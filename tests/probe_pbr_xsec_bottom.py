# krx_fundamental_safe.py
# ─────────────────────────────────────────────────────────────────────────
# 견고한 횡단면 fundamental 페처 + 리밸 재스캔
#
# 왜: Gate ① 결과 — get_market_fundamental 이 특정 날짜에 blank("")를 주고
#     pykrx가 그걸 0으로 치환(wrap.py:249)한다. 6월 말/12월 말을 하드코딩하면
#     그 리밸 연도 B/M이 통째로 비고 에러도 안 난다. cap 기준 거래일 판정은
#     cap↔fundamental 불일치로 못 막는다 → fundamental 자체의 비어있음을 기준으로 walk-back.
#
# 이 페처는 버리는 게 아니라 5a Extract 의 날짜 리졸버로 그대로 승격.
#
# 실행: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\tests\krx_fundamental_safe.py"
#       (경로는 compare 노트북 sys.executable 로 확인한 진짜 인터프리터)
# ─────────────────────────────────────────────────────────────────────────

# %% [0] 로그인(.env 먼저) + import + 경고 소거
import os, warnings
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)   # replace("",0) 다운캐스팅 경고, 무해

# %% [1] 견고한 페처 — fundamental 의 실제 채워짐을 기준으로 이전 방향 walk-back
def resolve_and_fetch(target, market, max_walk=15, min_count=50):
    """target(YYYYMMDD) 이전 방향으로 걸어가며 PBR>0 종목수 >= min_count 인
       첫 날짜의 fundamental 반환. blank(->0)·휴장·빈프레임은 자동으로 건너뜀.
       반환: dict(target, used, walked, n_valid, ok, frame)."""
    d = pd.Timestamp(target)
    for k in range(max_walk + 1):
        ds = d.strftime("%Y%m%d")
        try:
            f = stock.get_market_fundamental(ds, market=market)
        except Exception:
            f = None
        n = 0 if (f is None or len(f) == 0) else int((f["PBR"] > 0).sum())
        if n >= min_count:
            return {"target": target, "used": ds, "walked": k, "n_valid": n, "ok": True, "frame": f}
        d -= pd.Timedelta(days=1)
    return {"target": target, "used": None, "walked": max_walk, "n_valid": 0, "ok": False, "frame": None}

# %% [2] 재스캔 — June 리밸(size 6월) 2003~2012 · Dec 리밸(B/M 12월 ME) 2002~2011
#   walked 가 작으면(<= 한 주) 바닥 2003 확정 + Extract 는 walk-back 만 붙이면 끝.
#   특정 해가 ok=False 거나 walked 가 크면 그 해 횡단면이 진짜 희소 → by-ticker 조립 필요.
def scan(label, dates, market):
    rows = [resolve_and_fetch(t, market) for t in dates]
    df = pd.DataFrame(rows)[["target", "used", "walked", "n_valid", "ok"]]
    print(f"\n=== {label} ({market}) ===")
    print(df.to_string(index=False))
    bad = df[~df["ok"]]
    farr = df[df["walked"] > 7]
    if len(bad):
        print(f"  ⚠ 리졸브 실패(15일 내 유효 날짜 없음): {bad['target'].tolist()} → 그 해 by-ticker 조립 후보")
    if len(farr):
        print(f"  ⚠ walk-back 8일 이상: {list(zip(farr['target'], farr['walked']))} → 날짜 확인")
    if len(bad) == 0 and len(farr) == 0:
        print("  ✅ 전 리밸 walk-back 7일 이내로 해결 → 이 시장 바닥·엔드포인트 OK")
    return df

jun = [f"{y}0630" for y in range(2003, 2013)]
dec = [f"{y}1230" for y in range(2002, 2012)]

kospi_jun  = scan("June 리밸 (size)",       jun, "KOSPI")
kospi_dec  = scan("Dec 리밸 (B/M ME)",       dec, "KOSPI")
kosdaq_jun = scan("June 리밸 (size)",       jun, "KOSDAQ")
kosdaq_dec = scan("Dec 리밸 (B/M ME)",       dec, "KOSDAQ")

# %% [3] 요약 — 실제 표본 바닥 판정
print("\n" + "=" * 60)
print("판정")
print("=" * 60)
for name, dfj, dfd in [("KOSPI", kospi_jun, kospi_dec), ("KOSDAQ", kosdaq_jun, kosdaq_dec)]:
    okj = dfj[dfj["ok"]]["target"].min() if dfj["ok"].any() else None
    okd = dfd[dfd["ok"]]["target"].min() if dfd["ok"].any() else None
    print(f"{name}: 유효 첫 June={okj} · 유효 첫 Dec={okd}")
print("\n→ 모든 리밸이 ok=True·walk-back 소폭이면 PRD §2 바닥 2003 유지, D3·H3(2008 포함) 그대로.")
print("→ 특정 해 ok=False면 그 해만 by-ticker 조립 대상으로 §2 각주 + Extract 분기 설계.")
print("→ 어느 쪽이든 resolve_and_fetch 를 Extract 날짜 리졸버로 승격(리밸마다 used·walked 로깅).")