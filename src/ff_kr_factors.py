"""
Korean FF3 factor construction — 5b layer (on top of the LOCKED ff_core engine).

C0 (this module, so far): daily CD91 -> monthly risk-free.

Design (no look-ahead): the risk-free return earned over month M is fixed at the start of
month M, i.e. from the CD91 annual rate observed on the last trading day of month M-1. We
never average within the month and never peek at month M's own rates. The 91-day annual
rate r (percent) is converted to a monthly simple return by

    RF_M = (1 + r/100) ** (1/12) - 1

Calendar alignment: RF_M is labelled with the month-end of month M, matching the return
panel's month-end labels (panel `date` is the holding month-end). So the July return row
(labelled 07-31) pairs with RF_M computed from the June-end rate — start-of-month information
only. Holidays on the reference month-end are walked back to the latest earlier trading day.

Additive module: no existing E/T/L signature is touched (hard rule 1). Uses the existing E5
loader (ff_kr_extract.e5_riskfree) for raw daily CD91 — ECOS is not re-implemented here.

Limitation: CD91 is a 91-day (~quarterly) instrument; `** (1/12)` monthly de-annualisation
is an approximation to a true 1-month rate, consistent with common FF-KR practice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_kr_extract as E

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ---- panel column conventions (verified against the 5a parquet, §9 schema) ----
RET_COL = "ret_m"            # monthly holding-period return
WEIGHT_COL = "weight_vw"     # VW weight = June-sort ME (mktcap_6), constant within holding year
DATE_COL = "date"            # holding month-end (datetime64), matches rf index labels
PORT_2X3 = "port_2x3"        # values S/L S/M S/H B/L B/M B/H
PORT_5X5 = "port_5x5"        # values S{size5}B{bm5}, 25 cells
CAPEX_FLAG_VALUE = "share_change_capex"   # adj_flag value for capital-raise ME distortion


def build_monthly_riskfree(start: str = "19950101", end: str | None = None) -> pd.DataFrame:
    """
    Build the month-end risk-free series from the E5 daily CD91 feed.

    Returns a `date`-indexed frame (month-end DatetimeIndex) with columns:
        rf            monthly simple risk-free return  (1 + r/100)**(1/12) - 1
        cd91_annual   the source CD91 annual rate (percent) used for that month
        ref_date      the trading day the rate was read from (last obs <= prior month-end)
        ref_walk_days calendar days walked back from the prior month-end to ref_date (0 = none)

    RF_M (labelled month-end of month M) uses the CD91 rate as of the last trading day of
    month M-1 — no look-ahead. Coverage starts at the first month whose prior month has a
    CD91 observation (1995-02, since CD91 begins 1995-01-03).
    """
    daily = E.e5_riskfree(start=start, end=end).sort_index()
    if len(daily) == 0:
        raise RuntimeError("E5 CD91 daily feed returned no rows")

    # reference months = every month that has at least one daily CD91 obs; each such
    # reference month M-1 produces RF for the following month M.
    ref_months = pd.period_range(daily.index.min().to_period("M"),
                                 daily.index.max().to_period("M"), freq="M")
    rows = []
    for rm in ref_months:
        ref_end = rm.to_timestamp("M")            # last calendar day of the reference month
        sub = daily.loc[:ref_end]                 # walk-back: latest obs on/before month-end
        if len(sub) == 0:
            continue
        rate = float(sub["cd91"].iloc[-1])
        used = sub.index[-1]
        rf_month = (rm + 1).to_timestamp("M")     # RF applies to the NEXT month
        rows.append({
            "date": rf_month,
            "rf": (1.0 + rate / 100.0) ** (1.0 / 12.0) - 1.0,
            "cd91_annual": rate,
            "ref_date": used,
            "ref_walk_days": int((ref_end - used).days),
        })
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.index.name = "date"
    return out


def write_riskfree(rf: pd.DataFrame, name: str = "riskfree_monthly",
                   formats=("parquet", "csv")) -> dict:
    """Persist the monthly RF under results/ (parquet + csv), per the L-layer convention."""
    RESULTS_DIR.mkdir(exist_ok=True)
    written = {}
    if "parquet" in formats:
        p = RESULTS_DIR / f"{name}.parquet"
        try:
            rf.to_parquet(p)
            written["parquet"] = str(p)
        except Exception as e:  # pragma: no cover - pyarrow guard
            import warnings
            warnings.warn(f"parquet skipped ({e}); install pyarrow to enable")
    if "csv" in formats:
        p = RESULTS_DIR / f"{name}.csv"
        rf.to_csv(p, encoding="utf-8-sig")
        written["csv"] = str(p)
    return written


# ============================================================================
# 5b factor layer — C1 (MKT-RF), C2 (SMB/HML), C3 (25 ports), R (regression/GRS)
#
# Read-only over the 5a panels. VW uses `weight_vw` (6-June ME, constant within the
# holding year) per the 5a design — no same-period/look-ahead ME. The econometric
# engine (`ff_core`) is LOCKED and only consumed here, never modified.
# ============================================================================


def load_panel(which: str) -> pd.DataFrame:
    """Read a 5a panel parquet ('primary' or 'supplementary'), read-only."""
    fn = {"primary": "panel_primary.parquet",
          "supplementary": "panel_supplementary.parquet"}[which]
    df = pd.read_parquet(RESULTS_DIR / fn)
    return df


def load_riskfree() -> pd.Series:
    """Load the C0 monthly risk-free as a `date`-indexed Series (month-end labels)."""
    rf = pd.read_parquet(RESULTS_DIR / "riskfree_monthly.parquet")
    return rf["rf"].sort_index()


def _cell_returns(df: pd.DataFrame, port_col: str, weighting: str = "vw",
                  exclude_capex_month: bool = False):
    """
    Monthly per-cell portfolio returns and stock counts.

    Returns (ret, counts): both DataFrames indexed by `date`, columns = cell labels.
        vw : sum(ret_m * weight_vw) / sum(weight_vw) within (date, cell)
        ew : mean(ret_m) within (date, cell)
    counts is the raw stock count per (date, cell) BEFORE any weighting (and after the
    optional capex exclusion). A cell-month with no stock is NaN in `ret`.

    exclude_capex_month=True drops rows flagged `adj_flag == 'share_change_capex'` for the
    month(s) affected before the cell return is formed (6c-sensitivity switch; default False
    = capex rows included, which is the main result).
    """
    if exclude_capex_month:
        df = df[df["adj_flag"] != CAPEX_FLAG_VALUE]

    counts = (df.groupby([DATE_COL, port_col]).size()
                .unstack(port_col))
    if weighting == "vw":
        rw = df[RET_COL] * df[WEIGHT_COL]
        g = (df.assign(_rw=rw)
               .groupby([DATE_COL, port_col])
               .agg(_rw=("_rw", "sum"), _w=(WEIGHT_COL, "sum")))
        ret = (g["_rw"] / g["_w"]).unstack(port_col)
    elif weighting == "ew":
        ret = df.groupby([DATE_COL, port_col])[RET_COL].mean().unstack(port_col)
    else:
        raise ValueError(f"weighting must be 'vw' or 'ew', got {weighting!r}")
    return ret.sort_index(), counts.reindex(ret.index).sort_index()


def _market_return(df: pd.DataFrame, weighting: str = "vw") -> pd.Series:
    """Research-universe market proxy: whole-panel monthly return (VW by weight_vw, or EW).

    NOTE this is the FACTOR universe (financials / SPAC / non-positive-BE already removed by
    5a), not a total-market index — documented as the research-universe market proxy."""
    if weighting == "vw":
        rw = df[RET_COL] * df[WEIGHT_COL]
        g = (df.assign(_rw=rw).groupby(DATE_COL)
               .agg(_rw=("_rw", "sum"), _w=(WEIGHT_COL, "sum")))
        mkt = g["_rw"] / g["_w"]
    elif weighting == "ew":
        mkt = df.groupby(DATE_COL)[RET_COL].mean()
    else:
        raise ValueError(f"weighting must be 'vw' or 'ew', got {weighting!r}")
    return mkt.sort_index()


# ---- C1 : MKT - RF ---------------------------------------------------------

def build_market_factor(df: pd.DataFrame, rf: pd.Series, weighting: str = "vw") -> pd.Series:
    """MKT-RF = research-universe market return (VW/EW) minus the C0 monthly risk-free."""
    mkt = _market_return(df, weighting=weighting)
    aligned_rf = rf.reindex(mkt.index)
    return (mkt - aligned_rf).rename(f"MKT-RF_{weighting}")


# ---- C2 : SMB / HML from the 2x3 grid --------------------------------------

def build_smb_hml(df: pd.DataFrame, weighting: str = "vw",
                  exclude_capex_month: bool = False):
    """
    SMB / HML from the 6 `port_2x3` cells (Fama-French 1993 2x3 construction).

        SMB = mean(S/L, S/M, S/H) - mean(B/L, B/M, B/H)
        HML = mean(S/H, B/H) - mean(S/L, B/L)

    Returns (factors, cell_ret, cell_counts):
        factors    : date-indexed DataFrame [SMB, HML]
        cell_ret   : date-indexed DataFrame, the 6 cell returns
        cell_counts: date-indexed DataFrame, stock counts per cell
    """
    ret, counts = _cell_returns(df, PORT_2X3, weighting=weighting,
                                exclude_capex_month=exclude_capex_month)
    S = ret[["S/L", "S/M", "S/H"]].mean(axis=1)
    B = ret[["B/L", "B/M", "B/H"]].mean(axis=1)
    H = ret[["S/H", "B/H"]].mean(axis=1)
    L = ret[["S/L", "B/L"]].mean(axis=1)
    factors = pd.DataFrame({"SMB": S - B, "HML": H - L})
    return factors, ret, counts


# ---- C3 : 25 test portfolios (5x5) excess returns --------------------------

def build_25_excess(df: pd.DataFrame, rf: pd.Series, weighting: str = "vw"):
    """
    25 (5x5) test-portfolio excess returns: cell return (VW/EW) minus monthly RF.

    Returns (excess, cell_ret, cell_counts):
        excess     : date-indexed DataFrame, 25 columns (S1B1..S5B5), = cell_ret - rf
        cell_ret   : raw cell returns
        cell_counts: stock counts per cell-month
    """
    ret, counts = _cell_returns(df, PORT_5X5, weighting=weighting)
    # canonical column order S1B1..S5B5
    cols = [f"S{s}B{b}" for s in range(1, 6) for b in range(1, 6)]
    ret = ret.reindex(columns=cols)
    counts = counts.reindex(columns=cols)
    aligned_rf = rf.reindex(ret.index)
    excess = ret.sub(aligned_rf, axis=0)
    return excess, ret, counts


def factor_tstat(series: pd.Series) -> float:
    """Simple mean t-stat (mean / (sd/sqrt(T))), matching the report's factor-mean column."""
    x = series.dropna().values
    T = len(x)
    if T < 2:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(T)))


if __name__ == "__main__":
    rf = build_monthly_riskfree()
    paths = write_riskfree(rf)
    print(f"monthly RF: {len(rf)} rows  {rf.index.min().date()} .. {rf.index.max().date()}")
    print(f"written: {paths}")
