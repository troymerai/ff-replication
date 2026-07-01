"""Step 1 boundary pilots (HANDOVER 2 §4). Validates the sample-window / KOSDAQ-onset
skip logic that the 2010-06 pilot never exercised (2010 has both markets live).

Three rebalances via the existing L.run_rebalance:
    2003-06  (prior Dec 2002, KOSDAQ pre-onset)  -> KOSPI-only  (supplementary-window first rebalance)
    2005-06  (prior Dec 2004, KOSDAQ pre-onset)  -> KOSPI-only  (KOSDAQ skip fires)
    2006-06  (prior Dec 2005, KOSDAQ onset>=20051201) -> KOSPI+KOSDAQ (KOSDAQ enters)

Per rebalance, checks:
  - market composition matches the expected table (`market` column)
  - the KOSDAQ skip is a SKIP MARKER (fund.skip == 'kosdaq_pre_onset'), not an empty frame / error
  - two-ME integrity + §10 integrity pass
  - E3 cross-section / resolve works on old (2003) dates, incl. blank walk-back
  - E3 cross-sectional resolve for the return panel works that far back

Any mismatch -> FAIL line; the driver leaves a machine-checkable verdict at the end.
Output is written UTF-8 to results/boundary_pilots.txt (console is cp949).

Run: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\src\boundary_pilots.py"
"""
import io
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import ff_kr_extract as E
import ff_kr_transform as T
import ff_kr_load as L

OUT = io.open(ROOT / "results" / "boundary_pilots.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

CASES = [
    {"sort": "20030630", "expect_markets": {"KOSPI"},          "expect_kosdaq_skip": True},
    {"sort": "20050630", "expect_markets": {"KOSPI"},          "expect_kosdaq_skip": True},
    {"sort": "20060630", "expect_markets": {"KOSPI", "KOSDAQ"}, "expect_kosdaq_skip": False},
]

refs = T.reference_maps()
verdicts = []

for case in CASES:
    sort_date = case["sort"]
    dec = T.prior_december(sort_date)
    w("=" * 78)
    w(f"# boundary pilot — sort {sort_date} / prior Dec {dec}")

    # Inspect the raw E2 skip marker directly (not via the joined frame) ----------
    fund_kospi = E.e2_fundamentals(dec, "KOSPI")
    fund_kosdaq = E.e2_fundamentals(dec, "KOSDAQ")
    kosdaq_skip = fund_kosdaq.get("skip")
    w(f"  E2 KOSPI:  ok={fund_kospi['ok']} used={fund_kospi.get('used')} walked={fund_kospi.get('walked')}")
    w(f"  E2 KOSDAQ: ok={fund_kosdaq['ok']} skip={kosdaq_skip!r} frame_is_none={fund_kosdaq['frame'] is None}")

    # Full rebalance --------------------------------------------------------------
    r = L.run_rebalance(sort_date, refs=refs)
    sort, returns, panel, logs = r["sort"], r["returns"], r["panel"], r["logs"]
    integ = logs["integrity"]

    markets_present = set(sort["market"].unique()) if len(sort) else set()
    per_market = logs["meta"]["per_market"]
    w(f"  per-market meta:")
    for m, rec in per_market.items():
        w(f"    {m}: universe(afterT1)={rec.get('universe')} joined={rec.get('joined')} "
          f"skip={rec.get('skip')} bm_median={rec.get('bm_median')}")
    w(f"  sort rows={len(sort)} markets={sort['market'].value_counts().to_dict() if len(sort) else {}}")
    w(f"  E3 resolve (June cap6) used dates: "
      f"{sorted(sort['used_cap6_date'].dropna().unique().tolist()) if len(sort) else []}")
    w(f"  return panel: rows={len(returns)} tickers={returns['ticker'].nunique() if len(returns) else 0} "
      f"months={returns['date'].nunique() if len(returns) else 0}")
    if len(returns):
        w(f"    adj_flag={logs['adj_flag_counts']} implausible_dropped={len(logs['dropped_implausible'])} "
          f"panel_max|ret|={returns['ret_m'].abs().max():.3f}")
    w(f"  §10 integrity: {integ}")

    # Checks ----------------------------------------------------------------------
    checks = {}
    checks["market_composition"] = (markets_present == case["expect_markets"])
    if case["expect_kosdaq_skip"]:
        # skip marker present, frame None, NOT surfaced as an error/empty join treated as data
        checks["kosdaq_skip_marker"] = (kosdaq_skip == "kosdaq_pre_onset"
                                        and fund_kosdaq["frame"] is None
                                        and per_market.get("KOSDAQ", {}).get("skip") == "kosdaq_pre_onset")
        checks["kosdaq_absent_from_panel"] = ("KOSDAQ" not in markets_present)
    else:
        checks["kosdaq_included"] = ("KOSDAQ" in markets_present and kosdaq_skip is None)
    checks["integrity_passed"] = bool(integ.get("passed"))
    checks["two_me_present"] = (integ.get("two_me_present", 0) > 0)
    checks["e3_resolve_ok"] = (len(sort) > 0 and sort["used_cap6_date"].notna().all())
    checks["returns_nonempty"] = (len(returns) > 0)

    w(f"  CHECKS:")
    for k, v in checks.items():
        w(f"    [{'OK' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    verdicts.append((sort_date, ok, checks))
    w(f"  VERDICT {sort_date}: {'PASS' if ok else 'FAIL'}")

w("=" * 78)
allpass = all(v[1] for v in verdicts)
w(f"# BOUNDARY PILOTS OVERALL: {'ALL PASS' if allpass else 'FAIL — STOP'}")
for sd, ok, checks in verdicts:
    fails = [k for k, val in checks.items() if not val]
    w(f"  {sd}: {'PASS' if ok else 'FAIL ' + str(fails)}")
OUT.close()
print("wrote results/boundary_pilots.txt ; overall", "ALL PASS" if allpass else "FAIL")
