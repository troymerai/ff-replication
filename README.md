# Fama–French Three-Factor Replication — US Validation and Korean Extension

[한국어](README.ko.md) · **English**

A faithful replication of the Fama–French (1993) three-factor model (MKT–RF, SMB, HML), validated against Kenneth French's published US benchmarks and extended to the Korean equity market (KOSPI + KOSDAQ) using freely available data sources.

## Overview

The project has two aims: (1) validate a factor-construction and testing engine against the canonical US results as a correctness gate, and (2) apply the validated engine to Korean equities, characterizing how the factor structure differs from the US while being explicit about the biases that survivorship-incomplete free data introduce.

The emphasis is a faithful replication with transparent methodological choices and honest treatment of data limitations, rather than novel factor discovery.

**Status.** The US baseline engine is validated (below). The Korean data-availability groundwork is complete; the Korean ETL and factor construction are in progress. Korean factor results are forthcoming.

## Data sources

| Source | Provides | Coverage | Notes |
|---|---|---|---|
| Kenneth French Data Library | US 25 portfolios, factor series | public | Benchmark for engine validation; no raw data required |
| pykrx (KRX) | daily prices, market cap, listings, KRX P/B | market cap 1995-05+; cross-sectional P/B KOSPI 2002-12+, KOSDAQ 2005-12+; cross-sectional close prices both markets 2002-12+ | B/M source (via 1/PBR); requires a free KRX login |
| OpenDART | financial-statement items (book-equity components) | structured 2015+ | reserved for the "gold" book-equity loader (not used in the v0.1 B/M) |
| ECOS (Bank of Korea) | CD 91-day risk-free rate | 1995-01+ | series 817Y002 / item 010502000 |
| FinanceDataReader | delisting history | 1960–present | survivorship-bias tracking |

## Methodology

Three-factor construction follows Fama–French (1993):

- **Universe.** KOSPI + KOSDAQ common stocks; financials excluded (book-to-market is not comparable for leveraged / regulated-capital firms); KONEX excluded.
- **Accounting lag.** Prior fiscal-year book equity matched to July(t)–June(t+1) returns (≥ 6-month lag).
- **B/M.** BE(t−1) / ME(Dec t−1); size sorted on June ME (two distinct ME points). In the v0.1 pipeline B/M is taken from KRX-reported P/B (B/M ≈ 1/PBR); a gold book-equity loader from OpenDART is planned.
- **Weighting.** Value-weighted primary results; equal-weighted reported alongside as a micro-cap diagnostic.
- **Rebalancing.** Annual, end-of-June sort, July(t)–June(t+1) holding.

Rationale for each construction choice — alternatives considered and Korean-market context — is documented inline in the code and notebooks.

### Sample window

Cross-sectional P/B data begin at different dates by market: KOSPI 2002-12, KOSDAQ 2005-12 (the latter is a genuine data onset, confirmed at the security level, not an endpoint artifact). The common KOSPI+KOSDAQ three-factor series therefore begins at the June 2006 sort (returns from 2006-07). A KOSPI-only long panel (returns from 2003-07) is reported separately as a longer-horizon supplement rather than spliced into the primary series. MKT–RF alone can extend to 1995; SMB and HML require B/M and are bound by the windows above.

### Limitations (v0.1)

The replication is deliberately faithful in construction but transparent about the biases that free, survivorship-incomplete data introduce:

- **Returns are price returns, not dividend-inclusive total returns.** Fama–French uses total return. This is a conscious v0.1 simplification: high-dividend names are slightly under-measured, though the effect is largely offsetting within the long–short factors. A dividend (DPS) total-return path is planned for the gold phase.
- **Split/rights handling without a corporate-action feed.** Adjusted daily prices are unavailable before ~2014 from the free sources, so monthly returns are built from stitched month-end cross-sections (close × shares). On a material share-count change the return is taken from market-cap continuity (cap_ratio − 1), which is bounded and correct for splits, reverse-mergers, and bonus issues (the split-contaminated per-share price ratio is never used). For a genuine capital raise (rights offering, CB conversion) that same cap return **overstates** the holder return by the new-capital inflow — a bounded, logged approximation (`share_change_capex`), since rights value cannot be recovered without a corporate-action feed. Implausible months (|return| > 300%, chiefly halted micro-caps and distressed reverse-mergers) are dropped and logged.
- **B/M from KRX-reported P/B (B/M ≈ 1/PBR),** not a decomposed book-equity figure; an OpenDART book-equity loader is planned.
- **Survivorship.** Delisted names are held to their last trading day and dropped thereafter (no delisting-return adjustment, as that data is unavailable); results are framed as an upper bound on the survivorship-clean truth.
- **Point-in-time sector/classification approximation.** Financials are identified from the union of the current KRX-DESC industry labels and the delisting-frame industry labels, applied to the past (financial-sector membership is stable); financial holding companies are excluded while industrial holding companies are retained. Likewise, the fiscal year-end used for the accounting lag is the current settle month applied historically. Non-December firms (< 2% of listings) take their B/M from the KRX December cross-section rather than their own fiscal close — a small, known timing mismatch that is logged; an OpenDART book-equity loader fixes it in the gold phase.

## Repository structure

```
src/
  ff_core.py                 # GRS test + HAC (Newey–West) regression engine
  ff_data_us.py              # Kenneth French data loader
  validate_baseline.ipynb    # US baseline validation (25 regressions, GRS)
  ff_kr_extract.py           # Korea E (Extract): E1–E8 raw KRX/ECOS/FDR loaders + resolver
  ff_kr_transform.py         # Korea T (Transform): T1–T10 point-in-time clean panel
  ff_kr_load.py              # Korea L (Load): §9 monthly-long panel + parquet/sqlite/csv
  ff_kr_orchestrate.py       # full-period driver: per-year chunked, resumable (process-per-chunk)
  pilot_transform.py         # one-rebalance (2010-06) T+L pilot / validation harness
  boundary_pilots.py         # window/KOSDAQ-onset boundary validation (2003/2005/2006)
tests/
  validate_core_mc.ipynb     # Monte Carlo validation of the engine (size + power)
notebooks/
  pykrx_probe.ipynb                # KRX data-depth probe
  compare_krx_opendart_bm.ipynb    # KRX-P/B vs OpenDART-BE agreement check
  probe_ecos_krx_delisting.ipynb   # risk-free rate + delisting availability
probes/                        # KRX cross-section reliability checks
  probe_pbr_xsec_bottom.py       # cross-sectional P/B availability by date
  krx_fundamental_safe.py        # robust fetcher with blank-date walk-back
  check_bm_stability.py          # B/M-source temporal stability
  probe_kosdaq_byticker_2003.py  # KOSDAQ early-period data existence
  probe_price_depth.py           # cross-sectional close-price depth
```

## Reproduction

Requirements: Python 3.12; pykrx, OpenDartReader, FinanceDataReader, PublicDataReader, pandas, numpy, matplotlib, python-dotenv.

1. **Engine validation (no credentials).** Run `tests/validate_core_mc.ipynb` and `src/validate_baseline.ipynb`. The engine reproduces the published US 25-portfolio alphas and GRS statistic.
2. **Korean data access.** Requires a free KRX account and an OpenDART API key, supplied via a local `.env` (`KRX_ID`, `KRX_PW`, `DART_API_KEY`). See `.env.example`. The KRX session must be loaded before importing pykrx.
3. **Data-availability probes.** Run the scripts in `probes/` to reproduce the cross-section availability and B/M-agreement findings.

## Validation results

**Engine (Monte Carlo).** Empirical rejection rate 0.0500 at the 5% level under the null; power increases monotonically with injected alpha.

**US baseline (25 size–B/M portfolios, monthly).** Median regression R² = 0.926; small-growth alpha −0.674%/month (t = −4.71); joint GRS F(25, 1170) = 3.275, p = 1.18 × 10⁻⁷ — consistent with the published pattern of a rejected zero-alpha null driven by the small-growth corner.

## Data notes — KRX cross-section reliability

The KRX cross-sectional fundamental endpoint returns blank fields on scattered dates, which pykrx coerces to zero rather than raising an error. A naive fixed-date pull therefore risks silently empty cross-sections for whole rebalance years. The included fetcher resolves each rebalance date by walking back to the nearest date with populated fundamentals and logs the date actually used. KRX-P/B and OpenDART-derived book-to-market agree closely across sampled cross-sections (rank correlation ≈ 0.93–0.95; near-exact median book-value agreement), supporting the use of KRX-P/B as the v0.1 B/M source.

## Limitations

Free data sources do not include delisted-security returns. Korean results are therefore presented as a **survivorship-biased upper bound**, not as premium estimates: delistings concentrate in small, low-valuation (distressed) names, so their omission biases SMB and HML upward. The direction of the bias is argued qualitatively and supported by a low-delisting-window sensitivity; a full delisting-return correction is planned. Additional robustness items — liquidity filters, breakpoint universe, and cross-listing / holding-company adjustment (Korean dual-listed holding companies concentrate in the value bucket and can bias HML) — are documented as future work.

## References / license

Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56.

License: _to be added._
