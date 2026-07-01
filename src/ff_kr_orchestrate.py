"""
Korean FF3 full-period orchestrator (HANDOVER 2 §5) — chunked + resumable.

Builds the two point-in-time panels over the full sample:
    primary        KOSPI+KOSDAQ, sorts 2006-06 .. END   (KOSDAQ B/M available from 2006)
    supplementary  KOSPI-only,   sorts 2003-06 .. END   (longer one-market history)
The window switch is a single universe parameter — the `markets` tuple + start year per
panel (PANELS below). Nothing else in the T/L engine changes.

SESSION GUARD = process-per-chunk. pykrx logs into KRX at import, and the session expires
~1h later; a multi-hour single process would die mid-pull. So each rebalance-YEAR runs in a
fresh subprocess (`--worker`), which re-imports pykrx and gets a fresh KRX login. No re-login
hook in krx_call is needed — a new process is the cleanest renewal, and it is the natural
resume unit.

RESUME = per-year checkpoints (idempotent). For each (panel, year) the worker writes
    results/checkpoints/{panel}_{year}.parquet    (the §9 monthly-long panel for that sort)
    results/checkpoints/{panel}_{year}.json       (audit sidecar; written LAST = completion sentinel)
The driver skips any year whose .json sentinel already exists, so a crash/expiry resumes
from the first unfinished year and finished years are never recomputed. The parquet E-cache
(cache/*.parquet) additionally resumes *within* a year (cache hit -> no KRX call). Worker
writes parquet to a temp path then renames, so a half-written checkpoint is never mistaken
for complete.

OUTPUT: per-year checkpoints + merged panels results/panel_{primary,supplementary}.parquet
(+ .csv) + audit rollup results/orchestrate_report.txt / .json (exclusion counts, adj_flag
counts, delisting counts, implausible-dropped, integrity). The human-readable summary is
transcribed into CC_REPORT.md by hand (rule: recon/summary not inline).

USAGE
  driver (spawns one worker process per year, then merges):
    python ff_kr_orchestrate.py --panel primary       [--start-year 2006 --end-year 2025]
    python ff_kr_orchestrate.py --panel supplementary  [--start-year 2003 --end-year 2025]
    python ff_kr_orchestrate.py --panel both           [--start-year ... --end-year ...]
  merge only (rebuild merged panels + rollup from existing checkpoints):
    python ff_kr_orchestrate.py --merge-only --panel both
  worker (internal; one rebalance-year, fresh session):
    python ff_kr_orchestrate.py --worker --panel primary --year 2010
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
CKPT = RESULTS / "checkpoints"

# Panel definitions — the ONLY place the window/universe switch lives (§5).
PANELS = {
    "primary":       {"markets": ("KOSPI", "KOSDAQ"), "default_start": 2006, "breakpoint": "kospi_only"},
    "supplementary": {"markets": ("KOSPI",),          "default_start": 2003, "breakpoint": "kospi_only"},
}
DEFAULT_END = 2025   # last June sort with a complete 12-month holding window (data through 2026-06)


def _ckpt_paths(panel: str, year: int):
    return CKPT / f"{panel}_{year}.parquet", CKPT / f"{panel}_{year}.json"


def is_year_done(panel: str, year: int) -> bool:
    """A year is complete iff its JSON sentinel exists and reports status ok."""
    _, jp = _ckpt_paths(panel, year)
    if not jp.exists():
        return False
    try:
        return json.loads(jp.read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------- worker (one rebalance-year)
def run_worker(panel: str, year: int) -> int:
    """Run ONE rebalance in this (fresh-session) process and write its checkpoint.
    Returns process exit code (0 ok)."""
    import pandas as pd
    import ff_kr_transform as T
    import ff_kr_load as L

    spec = PANELS[panel]
    sort_date = f"{year}0630"
    CKPT.mkdir(parents=True, exist_ok=True)
    pq, jp = _ckpt_paths(panel, year)

    refs = T.reference_maps()
    r = L.run_rebalance(sort_date, markets=spec["markets"],
                        breakpoint_universe=spec["breakpoint"], refs=refs)
    panel_df, returns, logs = r["panel"], r["returns"], r["logs"]

    # write parquet atomically (temp -> rename) so a partial is never seen as complete
    tmp = pq.with_suffix(".parquet.tmp")
    panel_df.to_parquet(tmp, index=False)
    tmp.replace(pq)

    per_market = {m: {k: rec.get(k) for k in ("universe", "dropped", "joined", "skip", "bm_median")}
                  for m, rec in logs["meta"]["per_market"].items()}
    audit = {
        "panel": panel, "year": year, "sort_date": sort_date, "status": "ok",
        "markets_present": (sorted(panel_df["market"].unique().tolist())
                            if len(panel_df) and "market" in panel_df else []),
        "rows": int(len(panel_df)),
        "tickers": int(returns["ticker"].nunique()) if len(returns) else 0,
        "months": int(returns["date"].nunique()) if len(returns) else 0,
        "per_market": per_market,
        "integrity": logs["integrity"],
        "adj_flag_counts": logs["adj_flag_counts"],
        "implausible_dropped": int(len(logs["dropped_implausible"])),
        "delisted_rows": int(panel_df["is_delisted"].sum()) if len(panel_df) else 0,
        "breakpoints": {k: (v if not hasattr(v, "tolist") else v)
                        for k, v in logs["breakpoints"].items()},
        "panel_max_abs_ret": float(returns["ret_m"].abs().max()) if len(returns) else None,
    }
    jp.write_text(json.dumps(audit, ensure_ascii=False, default=float, indent=1), encoding="utf-8")
    print(f"[worker] {panel} {year}: rows={audit['rows']} markets={audit['markets_present']} "
          f"integrity_passed={logs['integrity'].get('passed')}")
    return 0


# ---------------------------------------------------------------- driver (spawn per year)
def run_driver(panels: list[str], start_year: int | None, end_year: int, spawn: bool = True) -> dict:
    """For each panel/year: skip if done, else spawn a fresh-process worker. Returns a summary."""
    CKPT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for panel in panels:
        spec = PANELS[panel]
        y0 = start_year if start_year is not None else spec["default_start"]
        years = list(range(y0, end_year + 1))
        done, skipped, failed = [], [], []
        for y in years:
            if is_year_done(panel, y):
                skipped.append(y)
                print(f"[driver] {panel} {y}: SKIP (checkpoint present)")
                continue
            if not spawn:                    # in-process (used by dry-run tests only)
                rc = run_worker(panel, y)
            else:
                t0 = time.time()
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__)), "--worker", "--panel", panel, "--year", str(y)],
                    cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
                dt = time.time() - t0
                rc = proc.returncode
                tail = (proc.stdout or "").strip().splitlines()[-1:] or (proc.stderr or "").strip().splitlines()[-3:]
                print(f"[driver] {panel} {y}: worker rc={rc} {dt:.1f}s | {' '.join(tail)}")
            (done if rc == 0 and is_year_done(panel, y) else failed).append(y)
        summary[panel] = {"years": years, "done": done, "skipped": skipped, "failed": failed}
    return summary


# ---------------------------------------------------------------- merge + audit rollup
def merge_panel(panel: str, start_year: int | None, end_year: int) -> dict:
    """Concatenate all present year checkpoints for a panel -> merged parquet+csv, and
    aggregate the per-year audit sidecars into one rollup dict."""
    import pandas as pd
    spec = PANELS[panel]
    y0 = start_year if start_year is not None else spec["default_start"]
    frames, audits, missing = [], [], []
    for y in range(y0, end_year + 1):
        pq, jp = _ckpt_paths(panel, y)
        if not (pq.exists() and jp.exists()):
            missing.append(y)
            continue
        frames.append(pd.read_parquet(pq))
        audits.append(json.loads(jp.read_text(encoding="utf-8")))
    if not frames:
        return {"panel": panel, "rows": 0, "missing_years": missing, "years": []}

    merged = pd.concat(frames, ignore_index=True).sort_values(["sort_date", "date", "ticker"]).reset_index(drop=True)
    out_pq = RESULTS / f"panel_{panel}.parquet"
    merged.to_parquet(out_pq, index=False)
    merged.to_csv(RESULTS / f"panel_{panel}.csv", index=False, encoding="utf-8-sig")

    # aggregate audit
    def _sum_flag(flag):
        return int(sum(a["adj_flag_counts"].get(flag, 0) for a in audits))
    rollup = {
        "panel": panel,
        "markets": list(spec["markets"]),
        "years": [a["year"] for a in audits],
        "missing_years": missing,
        "total_rows": int(len(merged)),
        "total_tickers": int(merged["ticker"].nunique()),
        "sort_dates": sorted(merged["sort_date"].unique().tolist()),
        "adj_flag_none": _sum_flag("none"),
        "adj_flag_split_bonus": _sum_flag("split_bonus"),
        "adj_flag_share_change_capex": _sum_flag("share_change_capex"),
        "implausible_dropped_total": int(sum(a["implausible_dropped"] for a in audits)),
        "delisted_rows_total": int(sum(a["delisted_rows"] for a in audits)),
        "panel_max_abs_ret": max((a["panel_max_abs_ret"] or 0) for a in audits),
        "integrity_all_passed": all(a["integrity"].get("passed") for a in audits),
        "integrity_fail_years": [a["year"] for a in audits if not a["integrity"].get("passed")],
        "per_year": [
            {"year": a["year"], "markets": a["markets_present"], "rows": a["rows"],
             "tickers": a["tickers"], "months": a["months"],
             "dropped": {m: rec.get("dropped") for m, rec in a["per_market"].items()},
             "implausible_dropped": a["implausible_dropped"],
             "integrity_passed": a["integrity"].get("passed")}
            for a in audits
        ],
        "out_parquet": str(out_pq),
    }
    return rollup


def write_rollup(rollups: list[dict], exclusions: dict | None):
    """Human-readable + JSON audit rollup under results/ (transcribe summary to CC_REPORT by hand)."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "orchestrate_audit.json").write_text(
        json.dumps({"panels": rollups, "exclusions": exclusions}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    out = io.open(RESULTS / "orchestrate_report.txt", "w", encoding="utf-8")
    def w(*a): print(*a, file=out)
    w("# Orchestrator audit rollup")
    if exclusions:
        w(f"\n## Universe exclusions (panel-invariant, from reference_maps)")
        w(f"  financial tickers identified (E7 KRX-DESC ∪ E6 delisting): {exclusions.get('financial_count')}")
    for r in rollups:
        w("=" * 74)
        w(f"## panel = {r['panel']}  markets={r.get('markets')}")
        if r.get("rows", r.get("total_rows", 0)) == 0:
            w(f"  (no checkpoints present; missing {r.get('missing_years')})"); continue
        w(f"  sorts: {r['sort_dates']}")
        w(f"  total rows {r['total_rows']:,} | unique tickers {r['total_tickers']:,} "
          f"| missing years {r['missing_years']}")
        w(f"  adj_flag: none {r['adj_flag_none']:,} / split_bonus {r['adj_flag_split_bonus']:,} "
          f"/ share_change_capex {r['adj_flag_share_change_capex']:,}")
        w(f"  implausible dropped (total) {r['implausible_dropped_total']} "
          f"| panel max|ret| {r['panel_max_abs_ret']:.3f} | delisted rows {r['delisted_rows_total']:,}")
        w(f"  integrity all passed: {r['integrity_all_passed']}  fail years: {r['integrity_fail_years']}")
        w(f"  per-year:")
        for py in r["per_year"]:
            w(f"    {py['year']}: markets={py['markets']} rows={py['rows']} tickers={py['tickers']} "
              f"months={py['months']} dropped={py['dropped']} implausible={py['implausible_dropped']} "
              f"integ={py['integrity_passed']}")
    out.close()
    print("wrote results/orchestrate_report.txt + orchestrate_audit.json")


def _exclusions_snapshot() -> dict:
    import ff_kr_transform as T
    refs = T.reference_maps()
    return {"financial_count": int(len(refs[3]))}


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["primary", "supplementary", "both"], default="both")
    ap.add_argument("--year", type=int, help="worker mode: single rebalance year")
    ap.add_argument("--worker", action="store_true", help="internal: run one year in this process")
    ap.add_argument("--start-year", type=int, default=None)
    ap.add_argument("--end-year", type=int, default=DEFAULT_END)
    ap.add_argument("--merge-only", action="store_true", help="rebuild merged panels from checkpoints")
    ap.add_argument("--no-spawn", action="store_true", help="run workers in-process (tests only)")
    args = ap.parse_args()

    if args.worker:
        if args.year is None:
            print("--worker requires --year", file=sys.stderr); sys.exit(2)
        sys.exit(run_worker(args.panel, args.year))

    panels = ["primary", "supplementary"] if args.panel == "both" else [args.panel]

    if not args.merge_only:
        summ = run_driver(panels, args.start_year, args.end_year, spawn=not args.no_spawn)
        print("[driver] summary:", json.dumps(summ))

    rollups = [merge_panel(p, args.start_year, args.end_year) for p in panels]
    write_rollup(rollups, _exclusions_snapshot() if not args.merge_only else None)


if __name__ == "__main__":
    main()
