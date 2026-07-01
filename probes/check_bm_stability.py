# check_bm_stability.py
# ─────────────────────────────────────────────────────────────────────────
# Validation check — B/M source point-in-time stability (KRX PBR vs OpenDART BE)
#
# Purpose: the primary B/M comes from KRX PBR (B/M = 1/PBR). This script checks
# that it agrees with book equity (BE) independently pulled from OpenDART, and
# that the agreement is stable across different points in time. Runs three points
# (2019/FY2018, 2021/FY2020, 2016/FY2015) at once and prints a cross-year table.
# Wraps the compare_krx_opendart_bm notebook logic in run_year(); extract_be is unchanged.
#
# Run: python probes/check_bm_stability.py
#   (three points x ~150 stocks x up to 2 DART calls -> a few minutes.)
# ─────────────────────────────────────────────────────────────────────────

# %% [0] setup — load .env first, then pykrx (KRX login), OpenDartReader
import os, io, time, contextlib, warnings
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
import OpenDartReader
from pykrx import stock
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

DART_KEY = os.environ.get("DART_API_KEY")
assert DART_KEY, "DART_API_KEY not found in .env."
dart = OpenDartReader(DART_KEY)
print("setup ok · DART key len:", len(DART_KEY))

# %% [1] helpers — same BE extraction as the notebook
#         (controlling-interest equity first, then total-equity fallback)
def to_num(x):
    s = str(x).replace(",", "").replace(" ", "").strip()
    if s in ("", "-", "--", "nan", "None"):
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v

def _finstate_silent(code, year, fs_div):
    """Call OpenDartReader quietly by suppressing its console logging."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return dart.finstate_all(code, year, reprt_code="11011", fs_div=fs_div)
    except Exception:
        return None

def extract_be(stock_code, year):
    """Controlling-interest equity first, total-equity fallback.
       Returns (value, fs_div, source_tag)."""
    for fs_div in ("CFS", "OFS"):
        df = _finstate_silent(stock_code, year, fs_div)
        if df is None or len(df) == 0:
            continue
        bs = df[df["sj_div"] == "BS"].copy()
        if bs.empty:
            continue
        bs["amt"] = bs["thstrm_amount"].map(to_num)
        nm  = bs["account_nm"].astype(str)
        aid = bs["account_id"].astype(str)
        c = bs[aid == "ifrs-full_EquityAttributableToOwnersOfParent"]           # controlling-interest equity (XBRL tag)
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "parent_id"
        c = bs[nm.str.contains("지배") & nm.str.contains("지분") & ~nm.str.contains("비지배")]  # controlling-interest equity (account name)
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "parent_nm"
        c = bs[aid == "ifrs-full_Equity"]                                       # total equity (XBRL tag)
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "total_id"
        c = bs[nm.str.contains("자본총계")]                                     # total equity (account name)
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "total_nm"
    return np.nan, None, None

# %% [2] run_year — compute B/M agreement metrics for one (DATE, FY)
def run_year(DATE, FY, MARKET="KOSPI", N_PER_Q=30, seed=42):
    fund  = stock.get_market_fundamental(DATE, market=MARKET)
    capdf = stock.get_market_cap(DATE, market=MARKET)
    krx = fund[["BPS", "PBR"]].join(capdf[["시가총액", "상장주식수"]], how="inner")
    krx = krx[(krx["PBR"] > 0) & (krx["시가총액"] > 0)].copy()
    assert len(krx) > 100, f"{DATE} cross-section looks empty (valid {len(krx)}) — check trading day / blank"
    krx["BM_krx"]         = 1.0 / krx["PBR"]
    krx["BE_krx_implied"] = krx["BPS"] * krx["상장주식수"]

    krx["Qsamp"] = pd.qcut(krx["BM_krx"], 5, labels=False)
    sample = krx.groupby("Qsamp", group_keys=False).sample(n=N_PER_Q, random_state=seed)

    recs = []
    for code in sample.index:
        be, fsd, how = extract_be(code, FY)
        recs.append({"ticker": code, "BE_dart": be, "fs_div": fsd, "how": how})
        time.sleep(0.03)
    be_df = pd.DataFrame(recs).set_index("ticker")

    m = sample.join(be_df[["BE_dart", "fs_div", "how"]], how="left")
    m = m[m["BE_dart"].notna() & (m["BE_dart"] > 0)].copy()
    m["BM_dart"]    = m["BE_dart"] / m["시가총액"]
    m["reldiff_BE"] = (m["BE_dart"] - m["BE_krx_implied"]) / m["BE_krx_implied"]

    rho = m["BM_krx"].corr(m["BM_dart"], method="spearman")
    lr  = np.log(m["BM_krx"]).corr(np.log(m["BM_dart"]))
    m["Qk"] = pd.qcut(m["BM_krx"], 5, labels=False)
    m["Qd"] = pd.qcut(m["BM_dart"], 5, labels=False)
    same_q  = (m["Qk"] == m["Qd"]).mean()
    within1 = (abs(m["Qk"] - m["Qd"]) <= 1).mean()
    med_abs = m["reldiff_BE"].abs().median()
    parent = m["how"].isin(["parent_id", "parent_nm"]).mean()   # share captured as controlling-interest equity

    return {"DATE": DATE, "FY": FY, "n": len(m),
            "spearman": round(rho, 3), "logBM_r": round(lr, 3),
            "same_q": round(same_q, 3), "within1": round(within1, 3),
            "med|reldiff|": round(med_abs, 3), "parent_share": round(parent, 3)}

# %% [3] loop over the three points + table
PAIRS = [("20190628", "2018"), ("20210628", "2020"), ("20160628", "2015")]
rows = []
for DATE, FY in PAIRS:
    print(f"... running {DATE} / FY{FY}")
    rows.append(run_year(DATE, FY))
tbl = pd.DataFrame(rows)
print("\n=== B/M source point-in-time stability ===")
print(tbl.to_string(index=False))
print("\nInterpretation: if Spearman stays near 0.9 at all three points and same_q >= 0.70, the B/M source is stable.")
print("If only one point drops sharply, check that year for preferred-stock / non-December fiscal-year-end skew. Low parent_share means BE is mostly the total-equity fallback (handled by the dedicated BE loader).")
