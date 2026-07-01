"""
Korean FF3 Load (L) layer.

Orchestrates one rebalance through the T layer and assembles the monthly-long point-in-time
panel (§9 schema), then persists it. One row per (holding-month, ticker), carrying the
end-of-June sort attributes (constant within the holding year) alongside that month's return
and the portfolio assignment — the direct input to factor construction (5b).

Storage (§9): parquet (bulk panel, needs pyarrow), sqlite (per-rebalance queries), csv
(human-readable snapshot). parquet is skipped with a warning if pyarrow is absent.

Panel columns (superset of §9 core; portfolio + weight columns added for 5b):
    date, ticker, market, mktcap_6, me_12, bps, pbr, bm, ret_m, adj_flag,
    is_delisted, is_holdco_dual, breakpoint_flag, used_fund_date, walked,
    size2, bm3, size5, bm5, port_2x3, port_5x5, weight_vw, sort_date
"""
from __future__ import annotations

import sqlite3
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ff_kr_transform as T

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_SORT_RENAME = {"me_6": "mktcap_6", "BPS": "bps", "PBR": "pbr", "walked_fund": "walked"}
PANEL_COLUMNS = [
    "date", "ticker", "market", "mktcap_6", "me_12", "bps", "pbr", "bm", "ret_m", "adj_flag",
    "is_delisted", "is_holdco_dual", "breakpoint_flag", "used_fund_date", "walked",
    "size2", "bm3", "size5", "bm5", "port_2x3", "port_5x5", "weight_vw", "sort_date",
]


def hold_window(sort_date: str) -> tuple[str, str]:
    """Holding window for a June-t sort: base = sort date, end = next June-end (t+1)."""
    d = pd.Timestamp(sort_date)
    return sort_date, f"{d.year + 1}0630"


def run_rebalance(sort_date: str,
                  markets=T.MARKETS,
                  breakpoint_universe: str = T.DEFAULT_BREAKPOINT_UNIVERSE,
                  refs=None) -> dict:
    """
    Full T pipeline for one rebalance. Returns dict(panel, sort, returns, breakpoints,
    logs) where `panel` is the monthly-long §9 frame and `logs` bundles the audit trail.
    """
    if refs is None:
        refs = T.reference_maps()
    built = T.build_sort_cross_section(sort_date, markets=markets, refs=refs)
    sort = built["frame"]
    if len(sort) == 0:
        return {"panel": pd.DataFrame(columns=PANEL_COLUMNS), "sort": sort,
                "returns": pd.DataFrame(), "breakpoints": {}, "logs": built["meta"]}

    asg = T.assign_portfolios(sort, breakpoint_universe=breakpoint_universe)
    adf = asg["frame"]

    start, end = hold_window(sort_date)
    returns = T.build_returns(adf.index.tolist(), start, end, markets=markets, delist=refs[1])

    panel = assemble_panel(adf, returns, sort_date)
    logs = {
        "meta": built["meta"],
        "breakpoints": asg["breakpoints"],
        "cell_counts": asg["cell_counts"],
        "integrity": T.integrity_checks(sort),
        "dropped_implausible": returns.attrs.get("dropped_implausible", pd.DataFrame()),
        "adj_flag_counts": (returns["adj_flag"].value_counts().to_dict() if len(returns) else {}),
    }
    return {"panel": panel, "sort": adf, "returns": returns,
            "breakpoints": asg["breakpoints"], "logs": logs}


def assemble_panel(sort_df: pd.DataFrame, returns: pd.DataFrame, sort_date: str) -> pd.DataFrame:
    """Join monthly returns to the constant end-of-June sort attributes -> §9 monthly-long panel."""
    if len(returns) == 0:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    attrs = sort_df.rename(columns=_SORT_RENAME).copy()
    attrs["weight_vw"] = attrs["mktcap_6"]
    keep = ["market", "mktcap_6", "me_12", "bps", "pbr", "bm", "is_holdco_dual",
            "breakpoint_flag", "used_fund_date", "walked", "size2", "bm3", "size5", "bm5",
            "port_2x3", "port_5x5", "weight_vw"]
    attrs = attrs[keep]
    panel = returns.merge(attrs, left_on="ticker", right_index=True, how="inner")
    panel["sort_date"] = sort_date
    return panel.reindex(columns=PANEL_COLUMNS).sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------- writers
def write_panel(panel: pd.DataFrame, name: str, formats=("csv", "sqlite", "parquet")) -> dict:
    """Persist the panel under results/ in the requested formats. Returns written paths."""
    RESULTS_DIR.mkdir(exist_ok=True)
    written = {}
    if "csv" in formats:
        p = RESULTS_DIR / f"{name}.csv"
        panel.to_csv(p, index=False, encoding="utf-8-sig")
        written["csv"] = str(p)
    if "sqlite" in formats:
        p = RESULTS_DIR / f"{name}.sqlite"
        con = sqlite3.connect(p)
        try:
            panel.assign(date=panel["date"].astype(str)).to_sql(
                "panel", con, if_exists="replace", index=False)
        finally:
            con.close()
        written["sqlite"] = str(p)
    if "parquet" in formats:
        p = RESULTS_DIR / f"{name}.parquet"
        try:
            panel.to_parquet(p, index=False)
            written["parquet"] = str(p)
        except Exception as e:
            warnings.warn(f"parquet skipped ({e}); install pyarrow to enable")
    return written
