#!/usr/bin/env python3
"""us_benchmark.py — H1·H2 비교 기준용 미국 팩터 통계.

Ken French의 3팩터 월별 자료(F-F_Research_Data_Factors)에서 세 구간의
MKT-RF·SMB·HML 평균, 표준편차, Newey-West HAC t를 계산한다.

  (1) 장기 앵커         : 1963-07 ~ 자료 최신월
  (2) 주 패널과 같은 창  : 2006-07 ~ min(자료 최신월, 2026-06)
  (3) 보조 패널과 같은 창 : 2003-07 ~ min(자료 최신월, 2026-06)

HAC 시차는 엔진(ff_core._nw_lag)을 그대로 가져와 쓴다 — 규칙의 단일 출처 유지.
(ff_core를 찾지 못하면 동일 공식 floor(4*(T/100)^(2/9))으로 폴백.)

위치·실행 (저장소 규약):
    src/us_benchmark.py 로 두고, 어느 폴더에서든
    python src/us_benchmark.py [KF_CSV_또는_ZIP_경로]

경로 인자를 주지 않으면 data/ 와 저장소 루트에서 F-F_Research_Data_Factors
파일을 찾고(ff_data_us.find_files와 같은 위치), 없으면 data/에 내려받는다.

산출:
  - results/us_benchmark.csv  (구간 x 팩터 통계)
  - 화면 출력: 통계 표 + 보고서 '표 1 주'에 그대로 붙일 교체 문장
  - results/factors_prim_vw.csv 가 있으면 같은 창으로 자른 한국 팩터를
    함께 출력해 대조를 확인한다(한국 자료는 소수 단위이므로 x100).
"""

import re
import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "src" else HERE
RESULTS = ROOT / "results"
sys.path.insert(0, str(HERE))

KF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
          "ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")

WINDOWS = [
    ("장기 앵커",        196307, None),
    ("주 패널과 같은 창",  200607, 202606),
    ("보조 패널과 같은 창", 200307, 202606),
]

# ----------------------------------------------------------------------
# HAC 시차: 엔진과 단일 출처
# ----------------------------------------------------------------------
try:
    from ff_core import _nw_lag as nw_lag  # noqa: E402
except Exception:
    def nw_lag(T: int) -> int:
        """Newey-West (1994) automatic lag rule: floor(4 * (T/100)^(2/9))."""
        return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def nw_stats(x: np.ndarray):
    """평균, 표준편차, HAC t (Bartlett 커널)."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    mu = x.mean()
    e = x - mu
    L = nw_lag(T)
    s = (e @ e) / T
    for l in range(1, L + 1):
        gamma = (e[l:] @ e[:-l]) / T
        s += 2.0 * (1.0 - l / (L + 1.0)) * gamma
    se = np.sqrt(s / T)
    return mu, x.std(ddof=1), mu / se, T, L


# ----------------------------------------------------------------------
# Ken French 자료 읽기 (월별 블록만)
# ----------------------------------------------------------------------
def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            return z.read(z.namelist()[0]).decode("latin-1")
    return path.read_text(encoding="latin-1")


def _find_local() -> Path | None:
    pats = ["F-F_Research_Data_Factors*.zip", "F-F_Research_Data_Factors*.CSV",
            "F-F_Research_Data_Factors*.csv"]
    bases = [ROOT / "data", ROOT, Path.cwd() / "data", Path.cwd()]
    seen = set()
    for base in bases:
        base = base.resolve()
        if base in seen or not base.exists():
            continue
        seen.add(base)
        for pat in pats:
            hits = sorted(base.glob(pat))
            if hits:
                return hits[0]
    return None


def load_kf(path_arg: str | None) -> pd.DataFrame:
    if path_arg:
        path = Path(path_arg)
    else:
        path = _find_local()
        if path is None:
            dest = ROOT / "data"
            dest.mkdir(exist_ok=True)
            path = dest / "F-F_Research_Data_Factors_CSV.zip"
            print(f"[us_benchmark] 로컬 사본이 없어 내려받습니다 -> {path}")
            urllib.request.urlretrieve(KF_URL, path)
    print(f"[us_benchmark] 자료: {path}")
    text = _read_text(path)

    rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{6})\s*,", line)
        if not m:
            continue
        parts = [p.strip() for p in line.split(",")]
        ym = int(parts[0])
        if 192601 <= ym <= 209912 and len(parts) >= 4:
            rows.append((ym, float(parts[1]), float(parts[2]), float(parts[3])))
    df = pd.DataFrame(rows, columns=["ym", "MKT-RF", "SMB", "HML"])
    return df.drop_duplicates("ym").sort_values("ym").reset_index(drop=True)


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    us = load_kf(path_arg)
    latest = int(us["ym"].max())
    print(f"[us_benchmark] 미국 월별 자료: {us['ym'].min()} ~ {latest} ({len(us)}개월)\n")

    out, stats = [], {}
    for name, a, b in WINDOWS:
        end = latest if b is None else min(latest, b)
        w = us[(us["ym"] >= a) & (us["ym"] <= end)]
        for f in ["MKT-RF", "SMB", "HML"]:
            mu, sd, t, T, L = nw_stats(w[f].values)
            out.append({"window": name, "start": a, "end": end, "factor": f,
                        "mean_pct": round(mu, 3), "sd_pct": round(sd, 2),
                        "nw_t": round(t, 2), "T": T, "nw_lag": L})
            stats[(name, f)] = (mu, t)
        print(f"--- {name}  {a}~{end}  (T={T}, lag={L}) ---")
        for f in ["MKT-RF", "SMB", "HML"]:
            mu, t = stats[(name, f)]
            print(f"  {f:7s} mean={mu:+.2f}%/월   NW t={t:+.2f}")
        print()

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(out).to_csv(RESULTS / "us_benchmark.csv", index=False,
                             encoding="utf-8-sig")
    print(f"[us_benchmark] 저장: {RESULTS / 'us_benchmark.csv'}\n")

    # 보고서 '표 1 주' 교체 문장
    p_end = min(latest, 202606)
    ym_str = f"{str(p_end)[:4]}-{str(p_end)[4:]}"
    m = {f: stats[("주 패널과 같은 창", f)] for f in ["MKT-RF", "SMB", "HML"]}
    g = {f: stats[("장기 앵커", f)] for f in ["MKT-RF", "SMB", "HML"]}
    sentence = (
        f"참고: 본 복제와 같은 창(2006-07~{ym_str}, 미국 자료 가용 구간)의 "
        f"미국(Ken French) 팩터 평균은 "
        f"MKT−RF {m['MKT-RF'][0]:+.2f}(t = {m['MKT-RF'][1]:.2f}), "
        f"SMB {m['SMB'][0]:+.2f}(t = {m['SMB'][1]:.2f}), "
        f"HML {m['HML'][0]:+.2f}(t = {m['HML'][1]:.2f})%/월이며, "
        f"미국 장기(1963-07 이후)는 "
        f"MKT−RF {g['MKT-RF'][0]:+.2f}, SMB {g['SMB'][0]:+.2f}, "
        f"HML {g['HML'][0]:+.2f}%/월이다."
    )
    print("=" * 72)
    print("[표 1 주 교체 문장]")
    print(sentence)
    print("=" * 72)

    # (선택) 같은 창으로 자른 한국 팩터 — 대조 확인용
    kr_path = RESULTS / "factors_prim_vw.csv"
    if kr_path.exists():
        kr = pd.read_csv(kr_path)
        kr["ym"] = pd.to_datetime(kr["date"]).dt.strftime("%Y%m").astype(int)
        w = kr[(kr["ym"] >= 200607) & (kr["ym"] <= p_end)]
        print("\n[대조] 같은 창으로 자른 한국(주 패널 VW) — 표 1 전 구간 값과의 차이 확인용")
        for f in ["MKT-RF", "SMB", "HML"]:
            mu, sd, t, T, L = nw_stats(w[f].values * 100.0)
            print(f"  {f:7s} mean={mu:+.2f}%/월   NW t={t:+.2f}   (T={T})")
        print("  ※ 미국 자료가 2026-06 이전에 끝나면 한두 달 차이는 생기며,"
              " 표 1의 전 구간 값과 사실상 같으면 그대로 보고해도 된다.")


if __name__ == "__main__":
    main()
