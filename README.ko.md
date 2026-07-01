# Fama–French 3팩터 복제 — 미국 검증과 한국 시장 확장

**한국어** · [English](README.md)

Fama–French(1993) 3팩터 모형(MKT–RF, SMB, HML)을 미국 시장에서 재현해 Kenneth French 공개값과 대조하고, 무료 데이터로 한국 시장(KOSPI + KOSDAQ)에 확장한 프로젝트.

## 개요

두 가지가 목표다. (1) 팩터 구성·검정 엔진을 미국 표준 결과로 검증해 정확성 게이트를 통과시키고, (2) 검증된 엔진을 한국 주식에 적용해 미국과의 팩터 구조 차이를 서술하되, 생존편향이 있는 무료 데이터가 낳는 편향을 명시적으로 다룬다.

새 팩터 발굴이 아니라 충실한 복제 + 투명한 방법론 선택 + 데이터 한계의 정직한 처리에 무게를 둔다.

**진행 상태.** 미국 baseline 엔진은 검증 완료(아래). 한국 데이터 가용성 기반 작업은 끝났고, 한국 ETL·팩터 구성은 진행 중이다. 한국 팩터 결과는 아직 산출 전이다.

## 데이터 소스

| 소스 | 제공 | 커버리지 | 비고 |
|---|---|---|---|
| Kenneth French Data Library | 미국 25 포트폴리오·팩터 시계열 | 공개 | 엔진 검증 기준값. raw 불필요 |
| pykrx (KRX) | 일별 가격·시총·종목리스트·KRX P/B | 시총 1995-05~, 횡단면 P/B KOSPI 2002-12~ / KOSDAQ 2005-12~, 횡단면 종가 양 시장 2002-12~ | B/M 소스(1/PBR). KRX 로그인 필요 |
| OpenDART | 재무제표 항목(자기자본 구성) | 구조화 2015+ | 골드 자기자본 로더용(v0.1 B/M엔 미사용) |
| ECOS (한국은행) | CD 91일 무위험 금리 | 1995-01~ | 통계표 817Y002 / 항목 010502000 |
| FinanceDataReader | 상장폐지 이력 | 1960~현재 | 생존편향 추적 |

## 방법론

3팩터 구성은 Fama–French(1993)를 따른다.

- **유니버스.** KOSPI + KOSDAQ 보통주. 금융주 제외(레버리지·규제자본 때문에 book-to-market 해석이 달라짐). KONEX 제외.
- **회계 lag.** 직전 회계연도 자기자본을 7월(t)~6월(t+1) 수익률에 매칭(≥ 6개월 lag).
- **B/M.** BE(t−1) / ME(12월 t−1); size는 6월 ME로 정렬(서로 다른 두 ME 시점). v0.1 파이프라인은 B/M을 KRX 공시 P/B에서 산출(B/M ≈ 1/PBR)하고, OpenDART 기반 골드 자기자본 로더는 별도 계획.
- **가중.** value-weighted를 주 결과로, equal-weighted를 마이크로캡 진단용으로 병기.
- **리밸런싱.** 연간, 6월 말 정렬, 7월(t)~6월(t+1) 보유.

각 구성 선택의 근거 — 검토한 대안과 한국 시장 맥락 — 은 코드·노트북에 함께 기록.

### 표본 창

횡단면 P/B 데이터의 시작 시점이 시장별로 다르다: KOSPI 2002-12, KOSDAQ 2005-12(후자는 종목 단위로 확인한 실제 데이터 온셋이며 엔드포인트 결함이 아니다). 따라서 KOSPI+KOSDAQ 공통 3팩터 시계열은 2006년 6월 정렬(2006-07 이후 수익률)부터 시작한다. KOSPI-only 장기 패널(2003-07 이후)은 주 시계열에 이어붙이지 않고 장기 조망용 보조로 별도 제시한다. MKT–RF만은 1995까지 확장 가능하나 SMB·HML은 B/M이 필요해 위 창에 묶인다.

## 레포 구조

```
src/
  ff_core.py                 # GRS 검정 + HAC(Newey–West) 회귀 엔진
  ff_data_us.py              # Kenneth French 데이터 로더
  validate_baseline.ipynb    # 미국 baseline 검증(25 회귀, GRS)
tests/
  validate_core_mc.ipynb     # 엔진 몬테카를로 검증(크기·검정력)
notebooks/
  pykrx_probe.ipynb                # KRX 데이터 깊이 프로브
  compare_krx_opendart_bm.ipynb    # KRX-P/B vs OpenDART-BE 일치도 검증
  probe_ecos_krx_delisting.ipynb   # 무위험 금리·상장폐지 가용성
probes/                        # KRX 횡단면 신뢰성 점검
  probe_pbr_xsec_bottom.py       # 날짜별 횡단면 P/B 가용성
  krx_fundamental_safe.py        # blank 날짜 walk-back 견고 페처
  check_bm_stability.py          # B/M 소스 시점 안정성
  probe_kosdaq_byticker_2003.py  # KOSDAQ 초기 구간 데이터 존재 여부
  probe_price_depth.py           # 횡단면 종가 깊이
```

## 재현

요구사항: Python 3.12; pykrx, OpenDartReader, FinanceDataReader, PublicDataReader, pandas, numpy, matplotlib, python-dotenv.

1. **엔진 검증(자격증명 불필요).** `tests/validate_core_mc.ipynb`와 `src/validate_baseline.ipynb` 실행. 엔진이 미국 25 포트폴리오 알파와 GRS 통계량을 재현한다.
2. **한국 데이터 접근.** 무료 KRX 계정과 OpenDART API 키가 필요하며, 로컬 `.env`(`KRX_ID`, `KRX_PW`, `DART_API_KEY`)로 공급한다. `.env.example` 참고. KRX 세션은 pykrx import보다 먼저 로드해야 한다.
3. **데이터 가용성 프로브.** `probes/`의 스크립트를 실행하면 횡단면 가용성·B/M 일치도 결과를 재현할 수 있다.

## 검증 결과

**엔진(몬테카를로).** 귀무가설하 5% 기각률 0.0500; 알파 주입 시 검정력이 단조 상승.

**미국 baseline(25 size–B/M 포트폴리오, 월별).** 회귀 R² 중앙값 0.926; 소형 성장주 알파 −0.674%/월(t = −4.71); 결합 GRS F(25, 1170) = 3.275, p = 1.18 × 10⁻⁷ — 소형 성장주 코너가 주도해 제로알파 귀무가설이 기각되는 공개 결과 패턴과 일치.

## 데이터 특이사항 — KRX 횡단면 신뢰성

KRX 횡단면 fundamental 엔드포인트는 특정 날짜에 빈 값을 반환하고, pykrx가 이를 에러 없이 0으로 치환한다. 그래서 날짜를 고정해 단순 조회하면 특정 리밸 연도의 횡단면이 조용히 비어버릴 위험이 있다. 포함된 페처는 각 리밸 날짜를 채워진 fundamental이 있는 가장 가까운 날짜로 walk-back해 해소하고, 실제 사용한 날짜를 로깅한다. KRX-P/B와 OpenDART 기반 book-to-market은 표본 횡단면 전반에서 근접하게 일치하며(순위상관 ≈ 0.93–0.95; 장부가 중앙값 거의 동일), KRX-P/B를 v0.1 B/M 소스로 쓰는 근거가 된다.

## 한계

무료 데이터 소스는 상장폐지 종목 수익률을 포함하지 못한다. 따라서 한국 결과는 프리미엄 추정치가 아니라 **생존편향으로 상방 편향된 상한선**으로 제시한다: 폐지 종목은 소형·저밸류(부실)에 몰려 있어 이들의 누락이 SMB·HML을 위로 편향시킨다. 편향의 방향은 정성적으로 논증하고 짧은 저폐지 구간 민감도로 뒷받침하며, 정식 폐지수익률 보정은 별도 계획이다. 추가 robustness 항목 — 유동성 필터, breakpoint 유니버스, 중복상장/지주사 보정(한국 동시상장 지주사는 가치 바구니에 몰려 HML을 편향시킬 수 있음) — 은 향후 과제로 기록.

## 참고문헌 / 라이선스

Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56.

라이선스: _추가 예정._
