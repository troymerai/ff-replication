"""
ff_data_us.py — 켄 프렌치 미국 데이터 읽기 (미국 전용)

하는 일:
  1) 켄 프렌치 압축파일/CSV에서 '월별' 구간만 떼어낸다
       - 3팩터 파일      : 월별 Mkt-RF, SMB, HML, RF
       - 25 포트폴리오 파일 : '월별 가치가중(Value Weighted)' 구간만
  2) 퍼센트 → 소수 (÷100)
  3) 빈 데이터(-99.99, -999) → NaN
  4) 두 파일의 날짜를 맞춘다 (공통 달만)
  5) 각 포트폴리오 수익에서 무위험이자(RF)를 빼 '초과수익'을 만든다

반환은 엔진(ff_core.summarize)이 그대로 받는 형태:
    excess_returns : (T x 25) 초과수익
    factors        : (T x 3)  [Mkt-RF, SMB, HML]
    rf             : (T,)     무위험이자 (참고용)

설계 메모: 줄 번호를 고정해두지 않는다. 켄 프렌치가 매달 새 줄을 덧붙여도
줄 번호가 밀리므로, '구간 제목'과 'YYYYMM 줄' 패턴만으로 구간을 찾는다.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# 켄 프렌치 직접 다운로드 주소 (로컬에서 새로 받을 때만 사용)
FACTORS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
               "ken.french/ftp/F-F_Research_Data_Factors_CSV.zip")
PORTFOLIOS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
                  "ken.french/ftp/25_Portfolios_5x5_CSV.zip")

_DATE_ROW = re.compile(r"^\s*\d{6}\s*,")   # 'YYYYMM,' 로 시작하는 월별 데이터 줄
_MISSING = [-99.99, -999]                  # 켄 프렌치 빈 데이터 표기


def _read_text(path: str) -> str:
    """경로가 .zip 이면 안의 CSV 하나를 읽고, 아니면 파일을 그대로 읽는다."""
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            return z.read(name).decode("latin-1")
    with open(path, "r", encoding="latin-1") as f:
        return f.read()


def _collect_block(raw_lines, header_idx):
    """칸 이름 줄(header_idx) 아래로 'YYYYMM,' 줄을 연속으로 모은다.
       데이터가 시작된 뒤 비데이터 줄(빈 줄·다음 구간 제목·연별 줄)을 만나면 멈춘다."""
    header_line = raw_lines[header_idx].strip()
    data = []
    for ln in raw_lines[header_idx + 1:]:
        if _DATE_ROW.match(ln):
            data.append(ln.strip())
        elif data:
            break
    return header_line, data


def _parse_block(header_line, data_lines):
    """칸 이름 줄 + 월별 데이터 줄들 → DataFrame.
       날짜는 월(Period) 인덱스, 값은 퍼센트→소수."""
    csv_text = header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text), skipinitialspace=True,
                     na_values=_MISSING)
    df = df.rename(columns={df.columns[0]: "date"})
    idx = pd.to_datetime(df["date"].astype(int).astype(str),
                         format="%Y%m").dt.to_period("M")
    df = df.drop(columns="date")
    df.index = idx
    df = df.replace(_MISSING, np.nan)        # 혹시 모를 잔여 표기 한 번 더
    return df / 100.0                        # 퍼센트 → 소수


def _load_factors(path: str):
    """3팩터 파일에서 월별 구간만. (factors[Mkt-RF,SMB,HML], rf) 반환."""
    raw = _read_text(path).replace("\r", "").split("\n")
    # 월별 칸 이름 줄 ',Mkt-RF,SMB,HML,RF' 를 찾는다 (그 위는 설명 글)
    h = next(i for i, ln in enumerate(raw) if ln.strip().startswith(",Mkt-RF"))
    header_line, data = _collect_block(raw, h)
    df = _parse_block(header_line, data)
    return df[["Mkt-RF", "SMB", "HML"]], df["RF"]


def _load_portfolios(path: str,
                     section="Average Value Weighted Returns -- Monthly"):
    """25 포트폴리오 파일에서 지정 구간(기본=월별 가치가중)만 떼어 DataFrame."""
    raw = _read_text(path).replace("\r", "").split("\n")
    s = next(i for i, ln in enumerate(raw) if section in ln)      # 구간 제목 줄
    h = next(i for i in range(s + 1, len(raw)) if raw[i].strip())  # 그 다음 칸 이름 줄
    header_line, data = _collect_block(raw, h)
    return _parse_block(header_line, data)


def load_ff_us(factors_path: str, portfolios_path: str,
               start: str | None = None, end: str | None = None):
    """
    켄 프렌치 미국 데이터를 읽어 엔진 입력 형태로 돌려준다.

    factors_path, portfolios_path : .zip 또는 .csv 경로
    start, end                    : 'YYYY-MM' 표본 구간 자르기 (선택)

    반환: (excess_returns[T x 25], factors[T x 3], rf[T])
    """
    factors, rf = _load_factors(factors_path)
    ports = _load_portfolios(portfolios_path)

    # 날짜 맞추기 (공통 달만)
    idx = factors.index.intersection(ports.index).sort_values()
    factors, rf, ports = factors.loc[idx], rf.loc[idx], ports.loc[idx]

    # 초과수익 = 포트폴리오 수익 − 무위험이자
    excess = ports.sub(rf, axis=0)

    # 표본 구간 자르기 (선택)
    if start is not None:
        excess, factors, rf = excess.loc[start:], factors.loc[start:], rf.loc[start:]
    if end is not None:
        excess, factors, rf = excess.loc[:end], factors.loc[:end], rf.loc[:end]

    return excess, factors, rf


def download_ff_us(dest_dir: str = "."):
    """(로컬 전용) 켄 프렌치에서 두 압축파일을 직접 받는다. 받은 경로 2개 반환.
       ※ 이 함수는 인터넷이 되는 네 PC에서만 동작한다."""
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for url in (FACTORS_URL, PORTFOLIOS_URL):
        fn = os.path.join(dest_dir, url.rsplit("/", 1)[1])
        urllib.request.urlretrieve(url, fn)
        paths.append(fn)
    return paths[0], paths[1]


def find_files(factors_name="F-F_Research_Data_Factors_CSV.zip",
               ports_name="25_Portfolios_5x5_CSV.zip"):
    """data/ 또는 현재 폴더에서 두 zip을 찾아 (factors_path, ports_path) 반환."""
    def _find(name):
        for d in [".", "data", os.path.join("..", "data")]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"'{name}' 를 못 찾음. data/ 폴더에 두세요.")
    return _find(factors_name), _find(ports_name)