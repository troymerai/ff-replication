# Fama–French 3요인 모형 복제 — 미국 검증과 한국 확장

**한국어** · [English](README.en.md)

Fama–French(1993) 3요인 모형(시장·규모·가치)을 무료 공개 데이터로 한국 주식시장(KOSPI+KOSDAQ)에 복제한 프로젝트다. 팩터를 계산하고 검정하는 코드를 답이 공개된 미국 데이터로 먼저 검증한 뒤, 같은 코드를 한국 시장에 적용한다.

무료 데이터에는 상장폐지된 종목의 수익률이 빠져 있어, 한국의 모든 수치는 실제보다 부풀려진 **상한값**으로 본다. 새로운 팩터를 발굴하기보다, 설계 결정을 투명하게 밝히고 데이터의 한계를 숨기지 않는 데 방점이 있다.

## 보고서

분석 배경·방법·결과·한계는 모두 보고서에 있다.

**[보고서 PDF](report/Replication of the Fama-French Three-Factor Model in the Korean Stock Market.pdf)**

## 저장소 구조

```
report/                      # 보고서 PDF (핵심 산출물)
src/
  ff_core.py                 # GRS 검정 + HAC 회귀 엔진
  ff_data_us.py              # 미국(Kenneth French) 데이터 로더
  validate_baseline.ipynb    # 미국 기준 검증
  ff_kr_extract.py           # 한국 데이터 수집
  ff_kr_transform.py         # 데이터 정제
  ff_kr_load.py              # 월별 패널 구성
  ff_kr_orchestrate.py       # 전 구간 실행 드라이버 (재개 가능)
  ff_kr_factors.py           # 팩터 + 25개 포트폴리오 구성
  ff_kr_analysis.py          # 회귀, 알파 표면, 시기별 분해
  us_benchmark.py            # 같은 창의 미국 팩터 평균
  regime_tests.py            # 구간 평균차 검정
tests/                       # 몬테카를로 엔진 검증 + pytest 스위트
notebooks/                   # 데이터 탐침 노트북
probes/                      # KRX 횡단면 신뢰성 점검
results/                     # 파이프라인 산출물 (감사 기록, 팩터, 회귀, 검정)
```

## 실행

요구사항: Python 3.12; pykrx, OpenDartReader, FinanceDataReader, PublicDataReader, pandas, numpy, matplotlib, python-dotenv.

1. **엔진 검증** `tests/validate_core_mc.ipynb`, `src/validate_baseline.ipynb`를 실행하면 미국 결과를 재현한다.
2. **한국 데이터 접근** 무료 KRX 계정과 OpenDART API 키가 필요하며, 로컬 `.env`(`KRX_ID`, `KRX_PW`, `DART_API_KEY`)로 공급한다. pykrx를 import하기 전에 KRX 세션을 로드해야 한다.
3. **전체 파이프라인** `src/ff_kr_orchestrate.py`로 패널과 팩터를 구성하고, `src/ff_kr_analysis.py`로 분석 산출물을 만든다.

## 라이선스

코드(`src/`, `tests/`, `probes/`, `notebooks/`)는 MIT 라이선스, 보고서와 문서(`report/`)는 CC BY 4.0을 따른다. 원자료(KRX, OpenDART, ECOS 등)의 권리는 각 제공처에 있으며, 본 저장소는 이를 내려받는 코드만 포함한다.
