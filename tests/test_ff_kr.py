"""
6a — engineering regression tests for the Korean FF3 replication (P4).

Locks the gates already passed by hand (5a §10 integrity, C0 risk-free, C1–R factor/GRS
construction, and the LOCKED ff_core engine golden) so they cannot silently regress.

Groups:
  A  factor math on a synthetic golden panel (build_smb_hml / _cell_returns / build_25_excess)
  B  C0 monthly risk-free output contract
  C  C1–C3 gate contract over the persisted 5b outputs
  D  R + engine: GRS dimensions, monotone loadings, ff_core US golden, determinism
  E  5a §10 panel integrity (pre-satisfied, pinned)

Read-only over results/*; synthetic fixtures are built in-test. Nothing is committed.
Run:  venv/Scripts/python -m pytest -q
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SRC = REPO / "src"

import sys
sys.path.insert(0, str(SRC))

import ff_core  # LOCKED engine — pure numpy/statsmodels, no network

# ff_kr_factors imports ff_kr_extract -> pykrx, which logs into KRX at import time.
# Guard so the suite degrades gracefully (skip factor-math) if creds/network are absent.
try:
    import ff_kr_factors as F
    _FACTORS_OK = True
    _FACTORS_ERR = ""
except Exception as e:  # pragma: no cover - environment dependent
    _FACTORS_OK = False
    _FACTORS_ERR = repr(e)

requires_factors = pytest.mark.skipif(
    not _FACTORS_OK, reason=f"ff_kr_factors import failed (KRX login?): {_FACTORS_ERR}")

# ---------------------------------------------------------------------------
# shared loaders (module-scoped so the parquets are read once)
# ---------------------------------------------------------------------------
EXPECTED_COLS = [
    "date", "ticker", "market", "mktcap_6", "me_12", "bps", "pbr", "bm", "ret_m",
    "adj_flag", "is_delisted", "is_holdco_dual", "breakpoint_flag", "used_fund_date",
    "walked", "size2", "bm3", "size5", "bm5", "port_2x3", "port_5x5", "weight_vw",
    "sort_date",
]
CELLS25 = [f"S{s}B{b}" for s in range(1, 6) for b in range(1, 6)]


@pytest.fixture(scope="module")
def panels():
    return {
        "primary": pd.read_parquet(RESULTS / "panel_primary.parquet"),
        "supplementary": pd.read_parquet(RESULTS / "panel_supplementary.parquet"),
    }


@pytest.fixture(scope="module")
def rf_series():
    return pd.read_parquet(RESULTS / "riskfree_monthly.parquet")


@pytest.fixture(scope="module")
def grs_summary():
    return json.loads((RESULTS / "grs_summary.json").read_text())


def _load_csv(name):
    df = pd.read_csv(RESULTS / f"{name}.csv")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


# ===========================================================================
# A. Factor math — synthetic golden panel
# ===========================================================================
def _mini_2x3():
    """One month, one stock per 2x3 cell, hand-set returns (VW == EW here)."""
    rets = {"S/L": 0.01, "S/M": 0.02, "S/H": 0.03, "B/L": 0.04, "B/M": 0.05, "B/H": 0.06}
    rows = []
    for cell, r in rets.items():
        size2, bm3 = cell.split("/")
        rows.append(dict(date=pd.Timestamp("2020-07-31"), ret_m=r, weight_vw=1_000,
                         port_2x3=cell, port_5x5="S1B1", size2=size2, bm3=bm3,
                         adj_flag="none"))
    return pd.DataFrame(rows)


@requires_factors
def test_A_smb_hml_golden():
    df = _mini_2x3()
    factors, cell, cnt = F.build_smb_hml(df, weighting="vw")
    # SMB = mean(SL,SM,SH) - mean(BL,BM,BH) = 0.02 - 0.05 = -0.03
    assert factors.loc["2020-07-31", "SMB"] == pytest.approx(-0.03, abs=1e-12)
    # HML = mean(SH,BH) - mean(SL,BL) = 0.045 - 0.025 = 0.02
    assert factors.loc["2020-07-31", "HML"] == pytest.approx(0.02, abs=1e-12)
    assert int(cnt.loc["2020-07-31"].sum()) == 6


@requires_factors
def test_A_cell_returns_vw_weighting():
    # cell S/L: two stocks, ret 0.10 (w=1) and 0.20 (w=3) -> VW = 0.70/4 = 0.175, EW = 0.15
    df = pd.DataFrame([
        dict(date=pd.Timestamp("2020-07-31"), ret_m=0.10, weight_vw=1, port_2x3="S/L", adj_flag="none"),
        dict(date=pd.Timestamp("2020-07-31"), ret_m=0.20, weight_vw=3, port_2x3="S/L", adj_flag="none"),
    ])
    vw, cnt = F._cell_returns(df, "port_2x3", weighting="vw")
    ew, _ = F._cell_returns(df, "port_2x3", weighting="ew")
    assert vw.loc["2020-07-31", "S/L"] == pytest.approx(0.175, abs=1e-12)
    assert ew.loc["2020-07-31", "S/L"] == pytest.approx(0.15, abs=1e-12)
    assert int(cnt.loc["2020-07-31", "S/L"]) == 2


@requires_factors
def test_A_build_25_excess_is_cell_minus_rf():
    df = pd.DataFrame([
        dict(date=pd.Timestamp("2020-07-31"), ret_m=0.05, weight_vw=10, port_5x5="S1B1", adj_flag="none"),
        dict(date=pd.Timestamp("2020-08-31"), ret_m=0.02, weight_vw=10, port_5x5="S1B1", adj_flag="none"),
    ])
    rf = pd.Series([0.01, 0.005],
                   index=pd.DatetimeIndex(["2020-07-31", "2020-08-31"], name="date"))
    excess, cell, cnt = F.build_25_excess(df, rf, weighting="vw")
    assert excess.loc["2020-07-31", "S1B1"] == pytest.approx(0.05 - 0.01, abs=1e-12)
    assert excess.loc["2020-08-31", "S1B1"] == pytest.approx(0.02 - 0.005, abs=1e-12)
    # 24 unpopulated cells are NaN this month
    assert excess.loc["2020-07-31", "S5B5"] != excess.loc["2020-07-31", "S5B5"]  # NaN


@requires_factors
def test_A_empty_and_single_stock_cells():
    # only S/L populated -> the other five 2x3 cells are absent (NaN) that month
    df = pd.DataFrame([
        dict(date=pd.Timestamp("2020-07-31"), ret_m=0.07, weight_vw=5, port_2x3="S/L", adj_flag="none"),
    ])
    vw, cnt = F._cell_returns(df, "port_2x3", weighting="vw")
    assert vw.loc["2020-07-31", "S/L"] == pytest.approx(0.07, abs=1e-12)   # single-stock cell
    assert "B/H" not in vw.columns or pd.isna(vw.reindex(columns=["B/H"]).iloc[0, 0])


@requires_factors
def test_A_exclude_capex_month():
    # S/L: normal 0.10 (w=1) + capex-flagged 0.50 (w=1)
    df = pd.DataFrame([
        dict(date=pd.Timestamp("2020-07-31"), ret_m=0.10, weight_vw=1, port_2x3="S/L",
             adj_flag="none"),
        dict(date=pd.Timestamp("2020-07-31"), ret_m=0.50, weight_vw=1, port_2x3="S/L",
             adj_flag="share_change_capex"),
    ])
    incl, _ = F._cell_returns(df, "port_2x3", weighting="vw", exclude_capex_month=False)
    excl, _ = F._cell_returns(df, "port_2x3", weighting="vw", exclude_capex_month=True)
    assert incl.loc["2020-07-31", "S/L"] == pytest.approx(0.30, abs=1e-12)   # (0.10+0.50)/2
    assert excl.loc["2020-07-31", "S/L"] == pytest.approx(0.10, abs=1e-12)   # capex dropped


# ===========================================================================
# B. C0 monthly risk-free contract
# ===========================================================================
def test_B_riskfree_no_missing_no_dup(rf_series):
    rf = rf_series
    assert rf["rf"].isna().sum() == 0
    assert rf.index.is_unique
    assert rf.index.is_monotonic_increasing


def test_B_riskfree_coverage_and_range(rf_series):
    rf = rf_series
    assert rf.index.min().year <= 1995            # coverage back to the mid-90s
    # monthly RF sane: strictly positive, well under 3%/month even in the 1998 IMF spike
    assert rf["rf"].min() > 0
    assert rf["rf"].max() < 0.03


def test_B_riskfree_no_lookahead(rf_series):
    rf = rf_series
    ref = pd.to_datetime(rf["ref_date"])
    label_month = rf.index.to_period("M")
    ref_month = ref.dt.to_period("M")
    # the rate used for month M is read in a strictly earlier month (start-of-month info)
    assert (ref_month.values < label_month.values).all()


def test_B_riskfree_walkback_bounded(rf_series):
    rf = rf_series
    within_panel = rf.loc[:"2026-06-30"]
    # holiday walk-backs inside the panel window are a few days at most
    assert within_panel["ref_walk_days"].max() <= 3


# ===========================================================================
# C. C1–C3 gate contract over persisted 5b outputs
# ===========================================================================
@pytest.mark.parametrize("tag", ["prim", "supp"])
def test_C_mktrf_alignment_and_premium(tag, rf_series):
    fac = _load_csv(f"factors_{tag}_vw")
    rf_idx = rf_series.index
    # every factor month maps 1:1 into the RF index (no off-by-one, 0 unmatched)
    assert fac.index.isin(rf_idx).all()
    assert fac["MKT-RF"].mean() > 0            # positive equity premium


@pytest.mark.parametrize("tag", ["prim", "supp"])
def test_C_smb_hml_signs_and_corr(tag):
    fac = _load_csv(f"factors_{tag}_vw")
    assert fac["HML"].mean() > 0                        # value premium positive
    assert fac["SMB"].mean() > 0                        # size premium positive (may be weak)
    corr = fac[["SMB", "MKT-RF"]].corr().iloc[0, 1]
    assert corr < 0                                     # SMB <-> MKT negative (KR literature)


@pytest.mark.parametrize("tag", ["prim", "supp"])
def test_C_2x3_cells_filled_every_month(tag):
    cells = _load_csv(f"cells2x3_{tag}_vw")
    # all 6 cells present and non-empty in every month
    assert list(cells.columns) == ["B/H", "B/L", "B/M", "S/H", "S/L", "S/M"] or set(cells.columns) == {
        "B/H", "B/L", "B/M", "S/H", "S/L", "S/M"}
    assert cells.notna().all().all()


@pytest.mark.parametrize("tag", ["prim", "supp"])
def test_C_25ports_common_sample_and_sparse_log(tag, grs_summary):
    excess = _load_csv(f"ports25_{tag}_vw")
    counts = _load_csv(f"ports25_counts_{tag}_vw")
    assert list(excess.columns) == CELLS25
    filled = excess.notna().all(axis=1)
    which = "primary" if tag == "prim" else "supplementary"
    # common (all-25-filled) sample length must match the T recorded for the GRS test
    assert int(filled.sum()) == grs_summary[which]["vw"]["T"]
    # sparse-cell log exists: at least one cell-month below 5 stocks (thin Big x High-BM corner)
    assert (counts < 5).to_numpy().sum() > 0


# ===========================================================================
# D. R + engine: GRS dims, monotone loadings, ff_core golden, determinism
# ===========================================================================
@pytest.mark.parametrize("which", ["primary", "supplementary"])
@pytest.mark.parametrize("w", ["vw", "ew"])
def test_D_grs_dimensions(which, w, grs_summary):
    g = grs_summary[which][w]
    assert g["N"] == 25 and g["K"] == 3
    T = g["T"]
    assert g["grs_dof"] == [25, T - 25 - 3]      # dof = (N, T-N-K) = (25, T-28)


@pytest.mark.parametrize("tag", ["prim", "supp"])
def test_D_loadings_monotone(tag):
    reg = pd.read_csv(RESULTS / f"regression_{tag}_vw.csv").set_index("portfolio")
    s_by_size = [reg.loc[[f"S{s}B{b}" for b in range(1, 6)], "beta_SMB"].mean() for s in range(1, 6)]
    h_by_bm = [reg.loc[[f"S{s}B{b}" for s in range(1, 6)], "beta_HML"].mean() for b in range(1, 6)]
    # s decreases with size, h increases with BM
    assert all(s_by_size[i] >= s_by_size[i + 1] for i in range(4)), s_by_size
    assert all(h_by_bm[i] <= h_by_bm[i + 1] for i in range(4)), h_by_bm


def _load_us():
    import ff_data_us
    ex, fac, rf = ff_data_us.load_ff_us(
        str(REPO / "data" / "F-F_Research_Data_Factors_CSV.zip"),
        str(REPO / "data" / "25_Portfolios_5x5_CSV.zip"))
    m = ex.notna().all(axis=1) & fac.notna().all(axis=1)
    return ex[m], fac[m]


def test_D_ffcore_us_golden():
    ex, fac = _load_us()
    _, grs, _ = ff_core.summarize(ex, fac, hac=True)
    # 6-pre US validation golden: F(25, 1170) ~ 3.2752 on the committed Ken French snapshot
    assert grs["dof1"] == 25 and grs["dof2"] == 1170
    assert grs["F"] == pytest.approx(3.275174, rel=1e-4)


def test_D_engine_determinism():
    ex, fac = _load_us()
    _, g1, _ = ff_core.summarize(ex, fac, hac=True)
    _, g2, _ = ff_core.summarize(ex, fac, hac=True)
    assert g1["F"] == g2["F"]
    assert g1["p_value"] == g2["p_value"]


# ===========================================================================
# E. 5a §10 panel integrity (pre-satisfied — pinned)
# ===========================================================================
@pytest.mark.parametrize("which", ["primary", "supplementary"])
def test_E_schema_23_cols(panels, which):
    df = panels[which]
    assert len(df.columns) == 23
    assert list(df.columns) == EXPECTED_COLS


@pytest.mark.parametrize("which", ["primary", "supplementary"])
def test_E_no_missing_key_fields(panels, which):
    df = panels[which]
    key = ["ret_m", "weight_vw", "mktcap_6", "bm", "port_2x3", "port_5x5",
           "size2", "bm3", "size5", "bm5"]
    assert df[key].isna().sum().sum() == 0


@pytest.mark.parametrize("which", ["primary", "supplementary"])
def test_E_positive_cap_and_bm(panels, which):
    df = panels[which]
    assert (df["mktcap_6"] > 0).all()
    assert (df["weight_vw"] > 0).all()
    assert (df["bm"] > 0).all()


@pytest.mark.parametrize("which", ["primary", "supplementary"])
def test_E_return_winsor_bound(panels, which):
    df = panels[which]
    assert df["ret_m"].abs().max() <= 2.996


@pytest.mark.parametrize("which", ["primary", "supplementary"])
def test_E_two_timepoint_me_identical_fraction(panels, which):
    df = panels[which]
    # June-sort ME (mktcap_6) vs prior-Dec ME (me_12): identical-value fraction must stay tiny
    frac = float((df["mktcap_6"] == df["me_12"]).mean())
    assert frac <= 0.021


@pytest.mark.parametrize("which", ["primary", "supplementary"])
def test_E_audit_columns_present(panels, which):
    df = panels[which]
    for col in ["adj_flag", "breakpoint_flag", "used_fund_date", "walked",
                "is_delisted", "is_holdco_dual"]:
        assert col in df.columns
