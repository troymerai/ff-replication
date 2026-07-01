# gate2_bm_stability.py
# ─────────────────────────────────────────────────────────────────────────
# Gate ② — D3 B/M 소스 시점 안정성 (KRX PBR ↔ OpenDART BE)
# 세 시점(2019/FY2018 · 2021/FY2020 · 2016/FY2015)을 한 번에 돌려 교차연도 표 출력.
# compare_krx_opendart_bm 노트북의 로직을 run_year()로 감싼 것. extract_be 는 그대로.
#
# 실행: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\tests\gate2_bm_stability.py"
#   (세 시점 × ~150 종목 × 최대 2 DART 호출 → 수 분 소요. 커널 재시작 후 처음부터.)
# ─────────────────────────────────────────────────────────────────────────

# %% [0] setup — .env 먼저, pykrx(KRX 로그인), OpenDartReader
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
assert DART_KEY, "DART_API_KEY가 .env에 없음."
dart = OpenDartReader(DART_KEY)
print("setup ok · DART key len:", len(DART_KEY))

# %% [1] helpers — 노트북과 동일한 BE 추출 (지배주주지분 우선 → 자본총계 폴백)
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
    """OpenDartReader의 콘솔 로그를 눌러서 조용히 호출."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return dart.finstate_all(code, year, reprt_code="11011", fs_div=fs_div)
    except Exception:
        return None

def extract_be(stock_code, year):
    """지배주주지분 우선, 자본총계 폴백. (값, fs_div, 출처태그) 반환."""
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
        c = bs[aid == "ifrs-full_EquityAttributableToOwnersOfParent"]           # 지배주주지분 태그
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "parent_id"
        c = bs[nm.str.contains("지배") & nm.str.contains("지분") & ~nm.str.contains("비지배")]  # 지배주주지분 계정명
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "parent_nm"
        c = bs[aid == "ifrs-full_Equity"]                                       # 자본총계 태그
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "total_id"
        c = bs[nm.str.contains("자본총계")]                                     # 자본총계 계정명
        if len(c) and pd.notna(c["amt"].iloc[0]):
            return c["amt"].iloc[0], fs_div, "total_nm"
    return np.nan, None, None

# %% [2] run_year — 한 (DATE,FY)의 B/M 일치 지표 산출
def run_year(DATE, FY, MARKET="KOSPI", N_PER_Q=30, seed=42):
    fund  = stock.get_market_fundamental(DATE, market=MARKET)
    capdf = stock.get_market_cap(DATE, market=MARKET)
    krx = fund[["BPS", "PBR"]].join(capdf[["시가총액", "상장주식수"]], how="inner")
    krx = krx[(krx["PBR"] > 0) & (krx["시가총액"] > 0)].copy()
    assert len(krx) > 100, f"{DATE} 횡단면이 빈 것 같음(유효 {len(krx)}) — 거래일/blank 확인"
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
    parent = m["how"].isin(["parent_id", "parent_nm"]).mean()   # 지배주주지분으로 잡힌 비율

    return {"DATE": DATE, "FY": FY, "n": len(m),
            "spearman": round(rho, 3), "logBM_r": round(lr, 3),
            "same_q": round(same_q, 3), "within1": round(within1, 3),
            "med|reldiff|": round(med_abs, 3), "parent_share": round(parent, 3)}

# %% [3] 세 시점 루프 + 표
PAIRS = [("20190628", "2018"), ("20210628", "2020"), ("20160628", "2015")]
rows = []
for DATE, FY in PAIRS:
    print(f"... running {DATE} / FY{FY}")
    rows.append(run_year(DATE, FY))
tbl = pd.DataFrame(rows)
print("\n=== D3 B/M 시점 안정성 ===")
print(tbl.to_string(index=False))
print("\n판정 가이드: Spearman 이 세 시점 모두 0.9 근처에서 안정 + same_q ≥ 0.70 유지면 D3 닫힘.")
print("한 시점만 크게 떨어지면 그해 우선주·비12월결산 편중 확인. parent_share 낮으면 BE가 자본총계 폴백 위주(D4 골드 로더 몫).")
