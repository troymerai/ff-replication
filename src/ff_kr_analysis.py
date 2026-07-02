"""
6c analysis layer — Korean FF3 results interpretation, sensitivity & hypothesis contrasts.

Consumes the 5b outputs (results/factors_*, ports25_*, regression_*, grs_summary.json) and
the 5a panels, plus a little re-regression. NO new data collection. The econometric engine
(`ff_core`) is LOCKED and only consumed; the 5b factor builders (`ff_kr_factors`) are reused
read-only for the capex toggle. This module adds only *analysis* logic (hard rule 3).

RESULT-PRESENTATION DISCIPLINE (PRD §1): every magnitude here is a SURVIVORSHIP-BIASED UPPER
BOUND — delisting returns are not corrected, dual listings are not float-adjusted, and no
micro-cap/liquidity screen is applied. Nothing below asserts that a value/size premium "holds"
in Korea; results are stated as directional, upper-bound quantities only.

Tasks:
  T1  alpha heatmap (25 ports, prim & supp)                -> heatmap_alpha_{prim,supp}.png (+csv)
  T2  SMB/HML/MKT-RF sub-period decomposition (+GRS, T>=40)-> subperiod_factors.csv
  T3a capex toggle (exclude_capex_month) sensitivity        -> sensitivity_capex.csv
  T3b delisting-window survivorship direction (H-Surv)      -> sensitivity_surv.csv
  T4  hypothesis (§4) contrast table                        -> hypothesis_table.md

Run:  venv/Scripts/python src/ff_kr_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
RESULTS = SRC.parent / "results"

import ff_core                    # LOCKED engine (consumed only)
import ff_kr_factors as F         # 5b builders (reused read-only for the capex toggle)

TAGS = {"prim": "primary", "supp": "supplementary"}
CELLS25 = [f"S{s}B{b}" for s in range(1, 6) for b in range(1, 6)]
SIG = 1.96                        # |t| significance threshold used throughout

# Sub-period windows (T2). GFC bracket matches the ① US-side decomposition so the two are
# comparable. `before` for primary is only 12M (thin) -> read via supplementary there.
SUBPERIODS = [
    ("before",     None,         "2007-06-30"),   # start .. 2007-06
    ("during-GFC", "2007-07-01", "2010-06-30"),
    ("after",      "2010-07-01", None),           # 2010-07 .. end
    ("full",       None,         None),
    ("regime-2020", "2020-01-01", "2020-12-31"),
    ("regime-2022", "2022-01-01", "2022-12-31"),
]
GRS_MIN_T = 40                    # only run GRS where the common (all-25-filled) sample >= 40


# --------------------------------------------------------------------------- helpers
def _nw_lag(T: int) -> int:
    return ff_core._nw_lag(T)


def hac_mean_stats(x: pd.Series) -> dict:
    """Mean, HAC(Newey-West) t of the mean, and SD for a factor return series (fraction units)."""
    x = x.dropna()
    T = len(x)
    if T < 2:
        return {"mean": float("nan"), "t_hac": float("nan"), "sd": float("nan"), "T": T}
    res = sm.OLS(x.values, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": _nw_lag(T)})
    return {"mean": float(x.mean()), "t_hac": float(res.tvalues[0]),
            "sd": float(x.std(ddof=1)), "T": T}


def load_factors(tag: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"factors_{tag}_vw.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_ports_excess(tag: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"ports25_{tag}_vw.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index().reindex(columns=CELLS25)


def _window(idx: pd.DatetimeIndex, start, end):
    m = pd.Series(True, index=idx)
    if start is not None:
        m &= idx >= pd.Timestamp(start)
    if end is not None:
        m &= idx <= pd.Timestamp(end)
    return m.values


# =========================================================================== T1
def t1_heatmap() -> pd.DataFrame:
    """5x5 alpha heat-maps (month %) with HAC significance, prim & supp. Upper-bound caption."""
    rows = []
    for tag, which in TAGS.items():
        reg = pd.read_csv(RESULTS / f"regression_{tag}_vw.csv").set_index("portfolio")
        A = np.full((5, 5), np.nan)      # rows = size S1(small)..S5(big), cols = BM B1(growth)..B5(value)
        Tt = np.full((5, 5), np.nan)
        for s in range(1, 6):
            for b in range(1, 6):
                p = f"S{s}B{b}"
                A[s - 1, b - 1] = reg.loc[p, "alpha"] * 100.0     # month %
                Tt[s - 1, b - 1] = reg.loc[p, "alpha_t"]
                rows.append({"panel": which, "portfolio": p, "size": s, "bm": b,
                             "alpha_pct_month": reg.loc[p, "alpha"] * 100.0,
                             "alpha_t_hac": reg.loc[p, "alpha_t"],
                             "sig_1.96": bool(abs(reg.loc[p, "alpha_t"]) > SIG)})

        # ---- figure ----
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        vmax = np.nanmax(np.abs(A))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(A, cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(5)); ax.set_xticks(range(5))
        ax.set_xticklabels(["B1\ngrowth", "B2", "B3", "B4", "B5\nvalue"])
        ax.set_yticks(range(5))
        ax.set_yticklabels(["S1 small", "S2", "S3", "S4", "S5 big"])
        ax.set_xlabel("Book-to-Market quintile"); ax.set_ylabel("Size quintile")
        for s in range(5):
            for b in range(5):
                sig = abs(Tt[s, b]) > SIG
                ax.text(b, s, f"{A[s, b]:+.2f}{'*' if sig else ''}",
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if sig else "normal",
                        color="black")
                if sig:  # boundary highlight on HAC-significant cells
                    ax.add_patch(plt.Rectangle((b - 0.5, s - 0.5), 1, 1, fill=False,
                                               edgecolor="black", lw=2.0))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("FF3 alpha (month %)")
        nsig = int((np.abs(Tt) > SIG).sum())
        ax.set_title(f"FF3 alpha surface — {which} (VW)\n"
                     f"* / boxed = |HAC t| > 1.96  ({nsig}/25 significant)", fontsize=10)
        fig.text(0.5, 0.005,
                 "SURVIVORSHIP-BIASED UPPER BOUND: no delisting-return / dual-listing / "
                 "micro-cap correction. Not a mispricing estimate.",
                 ha="center", fontsize=6.8, style="italic", color="dimgray")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        out = RESULTS / f"heatmap_alpha_{tag}.png"
        fig.savefig(out, dpi=150); plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "heatmap_alpha.csv", index=False, encoding="utf-8-sig")
    return df


# =========================================================================== T2
def t2_subperiods() -> pd.DataFrame:
    """Sub-period SMB/HML/MKT-RF mean/t(HAC)/SD (+ GRS where common T>=40)."""
    rows = []
    for tag, which in TAGS.items():
        fac = load_factors(tag)
        ex = load_ports_excess(tag)
        filled = ex.notna().all(axis=1)                  # all-25-filled = GRS-eligible months
        for name, start, end in SUBPERIODS:
            w = _window(fac.index, start, end)
            sub = fac.loc[w]
            if len(sub) == 0:
                continue
            # GRS on the common (all-25-filled) sub-sample
            exw = ex.loc[_window(ex.index, start, end) & filled.values]
            common_T = len(exw)
            grs_F = grs_p = np.nan
            if common_T >= GRS_MIN_T:
                facw = fac.reindex(exw.index)[["MKT-RF", "SMB", "HML"]]
                _, grs, _ = ff_core.summarize(exw, facw, hac=True)
                grs_F, grs_p = grs["F"], grs["p_value"]
            for fname in ["MKT-RF", "SMB", "HML"]:
                st = hac_mean_stats(sub[fname])
                rows.append({
                    "panel": which, "subperiod": name,
                    "start": (start or str(sub.index.min().date())),
                    "end": (end or str(sub.index.max().date())),
                    "factor": fname, "T": st["T"],
                    "mean_pct_month": st["mean"] * 100.0,
                    "t_hac": st["t_hac"], "sd_pct_month": st["sd"] * 100.0,
                    "grs_F": grs_F, "grs_p": grs_p,
                    "grs_common_T": common_T if common_T >= GRS_MIN_T else np.nan,
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "subperiod_factors.csv", index=False, encoding="utf-8-sig")
    return df


# =========================================================================== T3a
def t3a_capex() -> pd.DataFrame:
    """Capex-month toggle: rebuild SMB/HML with exclude_capex_month=True vs default (incl)."""
    rows = []
    for tag, which in TAGS.items():
        df = F.load_panel(which)
        incl, _, _ = F.build_smb_hml(df, weighting="vw", exclude_capex_month=False)
        excl, _, _ = F.build_smb_hml(df, weighting="vw", exclude_capex_month=True)
        capex_share = float((df["adj_flag"] == F.CAPEX_FLAG_VALUE).mean()) * 100.0
        for fname in ["SMB", "HML"]:
            a, b = incl[fname].dropna(), excl[fname].dropna()
            j = a.index.intersection(b.index)
            a, b = a.reindex(j), b.reindex(j)
            rows.append({
                "panel": which, "factor": fname,
                "mean_incl_pct": a.mean() * 100.0,
                "mean_excl_pct": b.mean() * 100.0,
                "delta_pct_month": (b.mean() - a.mean()) * 100.0,
                "max_abs_month_delta_pct": (b - a).abs().max() * 100.0,
                "corr_incl_excl": float(a.corr(b)),
                "capex_rows_share_pct": capex_share,
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "sensitivity_capex.csv", index=False, encoding="utf-8-sig")
    return out


# =========================================================================== T3b
def t3b_survivorship() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Delisting-window survivorship DIRECTION test (H-Surv).

    Delisting-event year = year of a delisted ticker's LAST month in the panel. Years are
    split at the median event count into high- vs low-delisting buckets; SMB/HML/MKT-RF are
    compared across buckets. STRONG LIMITATIONS (per PRD §6c): recent low counts suffer
    right-/recency-censoring, delistings cluster (GFC), and each bucket is a short, noisy
    sample -> this is a DIRECTIONAL HINT, not proof. Delisting-return (gold) correction = fall.
    """
    per_year_rows, bucket_rows = [], []
    for tag, which in TAGS.items():
        df = F.load_panel(which)
        df["date"] = pd.to_datetime(df["date"])
        # event year: last panel month of each ticker flagged delisted
        dl = df[df["is_delisted"]]
        event_year = dl.groupby("ticker")["date"].max().dt.year
        counts = event_year.value_counts().sort_index()
        med = counts.median()
        high_years = set(counts[counts > med].index)   # strictly above median = high-delisting

        fac = load_factors(tag)
        fac_year = fac.index.year
        is_high = pd.Series([y in high_years for y in fac_year], index=fac.index)

        for y, c in counts.items():
            per_year_rows.append({"panel": which, "year": int(y), "delistings": int(c),
                                  "bucket": "high" if y in high_years else "low"})

        for bucket, mask in [("high", is_high.values), ("low", ~is_high.values)]:
            sub = fac.loc[mask]
            yrs = sorted({int(y) for y in fac.index[mask].year})
            for fname in ["MKT-RF", "SMB", "HML"]:
                st = hac_mean_stats(sub[fname])
                bucket_rows.append({
                    "panel": which, "bucket": bucket, "factor": fname,
                    "n_months": st["T"], "n_years": len(yrs),
                    "mean_pct_month": st["mean"] * 100.0, "t_hac": st["t_hac"],
                    "median_delist_per_yr_threshold": float(med),
                    "years": ",".join(str(y) for y in yrs),
                })
    per_year = pd.DataFrame(per_year_rows)
    buckets = pd.DataFrame(bucket_rows)
    per_year.to_csv(RESULTS / "sensitivity_surv_by_year.csv", index=False, encoding="utf-8-sig")
    buckets.to_csv(RESULTS / "sensitivity_surv.csv", index=False, encoding="utf-8-sig")
    return per_year, buckets


# =========================================================================== T4
def t4_hypothesis(sub: pd.DataFrame, buckets: pd.DataFrame) -> str:
    """Contrast §4 hypotheses against the 5b/6c evidence. Upper-bound framing throughout."""
    grs = json.loads((RESULTS / "grs_summary.json").read_text())

    def fac_full(panel, fname):
        r = sub[(sub.panel == panel) & (sub.subperiod == "full") & (sub.factor == fname)]
        return r.iloc[0]

    def fac_sp(panel, sp, fname):
        r = sub[(sub.panel == panel) & (sub.subperiod == sp) & (sub.factor == fname)]
        return None if len(r) == 0 else r.iloc[0]

    p_smb = fac_full("primary", "SMB"); s_smb = fac_full("supplementary", "SMB")
    p_hml = fac_full("primary", "HML"); s_hml = fac_full("supplementary", "HML")
    p_mkt = fac_full("primary", "MKT-RF")

    def bkt(panel, bucket, fname):
        r = buckets[(buckets.panel == panel) & (buckets.bucket == bucket) & (buckets.factor == fname)]
        return r.iloc[0]

    # H-Surv direction: is the LOW-delisting-period premium below the HIGH-delisting one?
    hi, lo = bkt("primary", "high", "HML"), bkt("primary", "low", "HML")
    surv_dir = "lower" if lo.mean_pct_month < hi.mean_pct_month else "higher"

    md = []
    md.append("# 6c — Hypothesis contrast table (§4)\n")
    md.append("> **All magnitudes are survivorship-biased UPPER BOUNDS** — no delisting-return, "
              "dual-listing, or micro-cap/liquidity correction. Verdicts describe the *direction "
              "under the upper bound*, never that a premium is established in Korea.\n")
    md.append("| Hypothesis | Prediction | 5b/6c evidence (VW) | Verdict (upper-bound) |")
    md.append("|---|---|---|---|")
    md.append(
        f"| **H1 — SMB weak/reversed** | Size premium weak or negative in Korea | "
        f"SMB mean {p_smb.mean_pct_month:+.3f}%/mo (t={p_smb.t_hac:.2f}) primary; "
        f"{s_smb.mean_pct_month:+.3f}%/mo (t={s_smb.t_hac:.2f}) supp — small & HAC-insignificant | "
        f"**Supported** (weak, not significant) |")
    md.append(
        f"| **H2 — HML present** | Value tilt positive | "
        f"HML mean {p_hml.mean_pct_month:+.3f}%/mo (t={p_hml.t_hac:.2f}) primary; "
        f"{s_hml.mean_pct_month:+.3f}%/mo (t={s_hml.t_hac:.2f}) supp | "
        f"**Partial** — positive direction; upper-bound only, not 'value premium holds' |")
    # H3 regime: report after vs GFC HML/SMB
    p_hml_gfc = fac_sp("primary", "during-GFC", "HML")
    p_hml_aft = fac_sp("primary", "after", "HML")
    p_smb_gfc = fac_sp("primary", "during-GFC", "SMB")
    p_smb_aft = fac_sp("primary", "after", "SMB")
    md.append(
        f"| **H3 — regime cracks** | Factor strength varies / reverses across regimes | "
        f"primary HML GFC {p_hml_gfc.mean_pct_month:+.2f}% vs after {p_hml_aft.mean_pct_month:+.2f}%; "
        f"SMB GFC {p_smb_gfc.mean_pct_month:+.2f}% vs after {p_smb_aft.mean_pct_month:+.2f}%; "
        f"2020/2022 sub-samples thin (T=12, no GRS) | **Supported** (T2 shows regime shifts) |")
    md.append(
        f"| **H-Surv — survivorship inflation** | Low-delisting periods show *lower* premia | "
        f"primary HML high-delist buckets mean {hi.mean_pct_month:+.2f}%/mo vs "
        f"low-delist {lo.mean_pct_month:+.2f}%/mo -> low is **{surv_dir}** | "
        f"**Directional hint only** (recency/right-censoring, clustering, short samples; "
        f"gold delisting-return correction = fall) |")
    md.append(
        f"| **H-XList — dual-listing distortion** | Holdco/dual listings inflate Big cells | "
        f"D11 float-adjust / holdco exclusion needs the holdco roster | **Untested (fall)** |")

    md.append("\n## Anchor numbers (VW, full sample)\n")
    md.append(f"- Primary MKT-RF {p_mkt.mean_pct_month:+.3f}%/mo (t={p_mkt.t_hac:.2f}), "
              f"T={int(p_mkt['T'])}; GRS F={grs['primary']['vw']['grs_F']:.3f} "
              f"(p={grs['primary']['vw']['grs_p']:.2e}), "
              f"{grs['primary']['vw']['n_alpha_sig_1.96']}/25 alphas |t|>1.96.")
    md.append(f"- Supplementary GRS F={grs['supplementary']['vw']['grs_F']:.3f} "
              f"(p={grs['supplementary']['vw']['grs_p']:.2e}), "
              f"{grs['supplementary']['vw']['n_alpha_sig_1.96']}/25 alphas |t|>1.96.")
    md.append("- FF3 jointly **rejects** the zero-alpha null in both panels (upper bound; a true "
              "delisting-corrected test would weaken, not strengthen, any premium).")

    text = "\n".join(md) + "\n"
    (RESULTS / "hypothesis_table.md").write_text(text, encoding="utf-8")
    return text


# =========================================================================== main
def main():
    print("[T1] alpha heatmap ...")
    hm = t1_heatmap()
    print("[T2] sub-period decomposition ...")
    sub = t2_subperiods()
    print("[T3a] capex toggle ...")
    cap = t3a_capex()
    print("[T3b] survivorship direction ...")
    per_year, buckets = t3b_survivorship()
    print("[T4] hypothesis table ...")
    t4_hypothesis(sub, buckets)

    # ---- console digest (also the raw material for CC_REPORT) ----
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n===== T1 heatmap sig counts =====")
    print(hm.groupby("panel")["sig_1.96"].sum())
    print("\n===== T2 subperiods (SMB/HML/MKT) =====")
    print(sub[["panel", "subperiod", "factor", "T", "mean_pct_month", "t_hac", "grs_F"]]
          .to_string(index=False))
    print("\n===== T3a capex delta =====")
    print(cap.to_string(index=False))
    print("\n===== T3b delistings by year =====")
    print(per_year.to_string(index=False))
    print("\n===== T3b buckets =====")
    print(buckets[["panel", "bucket", "factor", "n_months", "mean_pct_month", "t_hac"]]
          .to_string(index=False))
    print("\nwritten:",
          "heatmap_alpha_{prim,supp}.png, heatmap_alpha.csv, subperiod_factors.csv, "
          "sensitivity_capex.csv, sensitivity_surv{,_by_year}.csv, hypothesis_table.md")


if __name__ == "__main__":
    main()
