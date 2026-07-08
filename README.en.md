# Fama–French Three-Factor Replication — US Validation and Korean Extension

[한국어](README.md) · **English**

A replication of the Fama–French (1993) three-factor model (market, size, value) applied to the Korean equity market (KOSPI + KOSDAQ) using freely available data. The factor-construction and testing code is first validated against the published US results, then applied to Korea.

Because free data omits delisted-security returns, every Korean number is treated as an **upper bound** on the true value. The emphasis is transparent design choices and honest treatment of data limits, not novel factor discovery.

## Report

Background, method, results, and limitations are all in the report (Korean).

**[Report PDF](report/Replication_of_the_Fama-French_Three-Factor_Model_in_the_Korean_Stock_Market.pdf)**

## Repository structure

```
report/                      # the report PDF (primary deliverable)
src/
  ff_core.py                 # GRS test + HAC regression engine
  ff_data_us.py              # US (Kenneth French) data loader
  validate_baseline.ipynb    # US baseline validation
  ff_kr_extract.py           # Korean data extraction
  ff_kr_transform.py         # data cleaning
  ff_kr_load.py              # monthly panel construction
  ff_kr_orchestrate.py       # full-period driver (resumable)
  ff_kr_factors.py           # factors + 25 portfolios
  ff_kr_analysis.py          # regressions, alpha surface, sub-period tables
  us_benchmark.py            # same-window US factor means
  regime_tests.py            # regime mean-equality tests
tests/                       # Monte Carlo engine validation + pytest suite
notebooks/                   # data-probe notebooks
probes/                      # KRX cross-section reliability checks
results/                     # pipeline outputs (audit log, factors, regressions, tests)
```

## Running

Requirements: Python 3.12; pykrx, OpenDartReader, FinanceDataReader, PublicDataReader, pandas, numpy, matplotlib, python-dotenv.

1. **Engine validation** Run `tests/validate_core_mc.ipynb` and `src/validate_baseline.ipynb` to reproduce the US results.
2. **Korean data access** Requires a free KRX account and an OpenDART API key via a local `.env` (`KRX_ID`, `KRX_PW`, `DART_API_KEY`). The KRX session must be loaded before importing pykrx.
3. **Full pipeline** `src/ff_kr_orchestrate.py` builds the panel and factors; `src/ff_kr_analysis.py` produces the analysis outputs.

## License

Code (`src/`, `tests/`, `probes/`, `notebooks/`) is released under the MIT License; the report and documentation (`report/`) under CC BY 4.0. Rights to the underlying data (KRX, OpenDART, ECOS, etc.) remain with their providers; this repository contains only the code to retrieve it.
