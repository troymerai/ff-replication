# probe_kosdaq_byticker_2003.py
# ─────────────────────────────────────────────────────────────────────────
# KOSDAQ by-ticker PBR 심도 — 2003~2005 조립 복구 가능성 판정
#
# 배경: krx_fundamental_safe 재스캔에서 KOSDAQ 횡단면은 2003~2005(Dec)·2003~2005(Jun)이
#       15일 창 통째로 빈다(ok=False). KOSPI는 횡단면 blank였어도 by-ticker로 2002까지
#       존재했다. KOSDAQ도 종목별로 존재하면 → 초기 구간만 by-ticker 조립해서
#       KOSPI+KOSDAQ를 2003부터 seam 없이 돌릴 수 있다(A 지배). 없으면 → A(2006~) 확정.
#
# 실행: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\tests\probe_kosdaq_byticker_2003.py"
# ─────────────────────────────────────────────────────────────────────────

# %% [0] 로그인(.env 먼저) + import
import os, warnings, random
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [1] 2003 시점 KOSDAQ 유니버스에서 표본 추출
def trading_day_on_or_before(yyyymmdd, max_back=10):
    d = pd.Timestamp(yyyymmdd)
    for _ in range(max_back + 1):
        ds = d.strftime("%Y%m%d")
        try:
            cap = stock.get_market_cap(ds, market="KOSDAQ")
        except Exception:
            cap = None
        if cap is not None and len(cap) > 0:
            return ds
        d = d - pd.Timedelta(1, "D")
    return None

td_base = trading_day_on_or_before("20031230")
univ = stock.get_market_ticker_list(td_base, market="KOSDAQ")   # 2003 시점 상장 KOSDAQ
random.seed(42)
probe_tickers = random.sample(univ, min(40, len(univ)))
print(f"2003 KOSDAQ 유니버스: {len(univ)}종목 · 프로브 표본: {len(probe_tickers)}")

# %% [2] by-ticker PBR 유효성 — 초기 구간(2003·2004·2005 각 6월/12월)
CHECK = ["2003-06", "2003-12", "2004-06", "2004-12", "2005-06", "2005-12"]
hit = {d: 0 for d in CHECK}
tot = 0
for t in probe_tickers:
    try:
        s = stock.get_market_fundamental_by_date("20030101", "20060101", t)
    except Exception:
        continue
    if s is None or len(s) == 0:
        continue
    tot += 1
    s.index = pd.to_datetime(s.index)
    for d in CHECK:
        yr, mo = map(int, d.split("-"))
        w = s[(s.index.year == yr) & (s.index.month == mo)]
        if len(w) and (w["PBR"] > 0).any():
            hit[d] += 1

print(f"\nby-ticker 응답 종목: {tot}/{len(probe_tickers)}")
for d in CHECK:
    share = f"{hit[d]}/{tot}" if tot else "0/0"
    print(f"  {d} PBR>0: {share}")

# %% [3] 판정
print("\n" + "=" * 60)
print("판정")
print("=" * 60)
if tot == 0:
    print("by-ticker 응답 0 — 종목별 엔드포인트도 KOSDAQ 초기를 못 준다. A(2006~) 확정.")
else:
    early = sum(hit[d] for d in ["2003-06", "2003-12", "2004-12"]) / (3 * tot)
    print(f"2003~2004 평균 유효율 ≈ {early:.0%}")
    print("  높으면(예 ≥ 60%) → KOSDAQ 2003~2005 by-ticker 조립 가능 → KOSPI+KOSDAQ 2003~ seam 없이(A 지배).")
    print("  낮거나 0이면      → KOSDAQ 초기 실재 부재 → A(2006~ 공통) 주 + KOSPI-only 2003~ 보조.")
print("\n※ 복구되더라도 조립 비용(종목별 pull)과 정합(횡단면↔by-ticker 값 일치)은 5a Extract에서 확인.")
