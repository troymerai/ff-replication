"""
Korean FF3 data extraction — E (Extract) layer.

Loaders: E1 universe · E2 fundamentals (B/M) · E3 market cap (size) ·
E4 prices (returns) · E5 risk-free (CD91) · E6 delisting.
Infra: KRX session guard, blank-date resolver (walk-back), parquet cache.

E does raw acquisition only. Filtering, normalization, and lag are the T layer.

Prereqs: .env at repo root with KRX_ID / KRX_PW / DART_API_KEY / ECOS_API_KEY.
pykrx logs in at import, so load_dotenv() must run first.
Optional: pip install pyarrow  (for the parquet cache; without it, calls just hit KRX every time)
"""

import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()                     # must precede pykrx import (import triggers KRX login)

import pandas as pd
from pykrx import stock

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------- config
MARKETS = ("KOSPI", "KOSDAQ")
KOSDAQ_BM_ONSET = "20051201"      # KOSDAQ cross-sectional B/M onset; earlier -> KOSPI-only
CACHE_DIR = Path(os.environ.get("FF_CACHE", "cache"))
CACHE_DIR.mkdir(exist_ok=True)
MIN_XSEC = 50                     # min populated rows to treat a cross-section as non-blank
ECOS_CD91_TABLE = "817Y002"       # ECOS CD 91-day
ECOS_CD91_ITEM = "010502000"


# ---------------------------------------------------------------- parquet cache
def _cache_path(key):
    return CACHE_DIR / f"{key}.parquet"

def cache_get(key):
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

def cache_set(key, df):
    try:
        df.to_parquet(_cache_path(key))
    except Exception as e:
        warnings.warn(f"cache write skipped for {key}: {e} (install pyarrow to enable)")


# ---------------------------------------------------------------- KRX session guard
# pykrx logs in at import (KRX_ID/KRX_PW in env); the session expires ~1h later.
# Short pulls are fine. For a full multi-hour panel pull, chunk the run so each
# chunk finishes within one session, or re-run in a fresh process per chunk.
def krx_call(fetch_fn, *args, retries=2, **kwargs):
    last = None
    for i in range(retries + 1):
        try:
            return fetch_fn(*args, **kwargs)
        except Exception as e:
            last = e
            time.sleep(1.0 * (i + 1))
    raise last


# ---------------------------------------------------------------- blank-date resolver
def resolve(target_date, fetch_fn, count_fn, min_count=MIN_XSEC, max_walk=15):
    """
    Walk back from target_date to the nearest date whose cross-section is populated.
    The KRX fundamental/ohlcv endpoints return blanks on scattered dates (pykrx
    coerces them to 0), so a fixed-date pull can silently yield an empty cross-section.
      fetch_fn(yyyymmdd) -> DataFrame | None
      count_fn(df)       -> int  (number of populated rows)
    Returns dict(used, walked, ok, frame).
    """
    d = pd.Timestamp(target_date)
    for k in range(max_walk + 1):
        ds = d.strftime("%Y%m%d")
        df = krx_call(fetch_fn, ds)
        n = 0 if (df is None or len(df) == 0) else count_fn(df)
        if n >= min_count:
            return {"used": ds, "walked": k, "ok": True, "frame": df}
        d = d - pd.Timedelta(days=1)
    return {"used": None, "walked": max_walk, "ok": False, "frame": None}


# ---------------------------------------------------------------- E1 universe
def e1_universe(date, market):
    """Raw listed tickers on `date` for `market` (point-in-time). Filtering is T-layer."""
    key = f"e1_universe_{market}_{date}"
    c = cache_get(key)
    if c is not None:
        return c["ticker"].tolist()
    tickers = krx_call(stock.get_market_ticker_list, date, market=market)
    cache_set(key, pd.DataFrame({"ticker": tickers}))
    return tickers


# ---------------------------------------------------------------- E2 fundamentals (B/M)
def _fund_fetch(market):
    return lambda ds: krx_call(stock.get_market_fundamental, ds, market=market)

def _fund_count(df):
    return int((df["PBR"] > 0).sum())

def e2_fundamentals(target_date, market):
    """
    Cross-sectional BPS/PBR at the resolved rebalance date (walk-back for blanks).
    KOSDAQ before its B/M onset returns a skip marker, not an error.
    frame columns: BPS, PER, PBR, EPS, DIV, DPS.
    """
    if market == "KOSDAQ" and target_date < KOSDAQ_BM_ONSET:
        return {"used": None, "walked": 0, "ok": False, "frame": None, "skip": "kosdaq_pre_onset"}
    key = f"e2_fund_{market}_{target_date}"
    c = cache_get(key)
    if c is not None:
        return {"used": target_date, "walked": 0, "ok": True, "frame": c}  # walk meta not cached
    r = resolve(target_date, _fund_fetch(market), _fund_count)
    if r["ok"]:
        cache_set(key, r["frame"])
    return r


# ---------------------------------------------------------------- E3 market cap (size)
def _cap_fetch(market):
    return lambda ds: krx_call(stock.get_market_cap, ds, market=market)

def _cap_count(df):
    return int((df["시가총액"] > 0).sum())

def e3_marketcap(target_date, market):
    """Cross-sectional market cap + shares outstanding at the resolved date."""
    key = f"e3_cap_{market}_{target_date}"
    c = cache_get(key)
    if c is not None:
        return {"used": target_date, "walked": 0, "ok": True, "frame": c}
    r = resolve(target_date, _cap_fetch(market), _cap_count)
    if r["ok"]:
        cache_set(key, r["frame"])
    return r


# ---------------------------------------------------------------- E4 prices (returns)
def _px_fetch(market):
    return lambda ds: krx_call(stock.get_market_ohlcv, ds, market=market)

def _px_count(df):
    col = "종가" if "종가" in df.columns else df.columns[3]
    return int((df[col] > 0).sum())

def e4_prices(target_date, market):
    """Cross-sectional OHLCV (month-end close) at the resolved date."""
    key = f"e4_px_{market}_{target_date}"
    c = cache_get(key)
    if c is not None:
        return {"used": target_date, "walked": 0, "ok": True, "frame": c}
    r = resolve(target_date, _px_fetch(market), _px_count)
    if r["ok"]:
        cache_set(key, r["frame"])
    return r


# ---------------------------------------------------------------- E5 risk-free (CD91)
def e5_riskfree(start="19950101", end=None):
    """
    ECOS CD 91-day daily risk-free (table 817Y002 / item 010502000, annual %).
    Ported from the working probe (probe_ecos_krx_delisting). Returns a normalized
    daily frame: DatetimeIndex `date` with a single `cd91` column (annual percent).
    Daily->monthly compounding is the 5b layer's job, not E's.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")
    key = f"e5_rf_cd91_{start}_{end}"
    c = cache_get(key)
    if c is not None:
        return c

    from PublicDataReader import Ecos
    ecos_key = os.environ.get("ECOS_API_KEY")
    if not ecos_key:
        raise RuntimeError("ECOS_API_KEY not set in .env")
    ecos = Ecos(ecos_key)
    s = krx_call(
        ecos.get_statistic_search,
        통계표코드=ECOS_CD91_TABLE, 주기="D",
        검색시작일자=start, 검색종료일자=end,
        통계항목코드1=ECOS_CD91_ITEM,
    )
    out = (pd.DataFrame({
        "date": pd.to_datetime(s["시점"], format="%Y%m%d"),
        "cd91": pd.to_numeric(s["값"], errors="coerce"),
    }).dropna().set_index("date").sort_index())
    cache_set(key, out)
    return out


# ---------------------------------------------------------------- E6 delisting
def e6_delisting():
    """
    Full KRX delisting history via FinanceDataReader ('KRX-DELISTING'), normalized to
    a 6-digit `ticker` index with parsed ListingDate/DelistingDate. Raw acquisition only:
    filtering to common stock / market / window is the T layer (T7). Cached to parquet.
    """
    key = "e6_delisting"
    c = cache_get(key)
    if c is not None:
        return c

    import FinanceDataReader as fdr
    raw = krx_call(fdr.StockListing, "KRX-DELISTING")
    df = raw.copy()
    df["DelistingDate"] = pd.to_datetime(df["DelistingDate"], errors="coerce")
    df["ListingDate"] = pd.to_datetime(df["ListingDate"], errors="coerce")
    df["ticker"] = df["Symbol"].astype(str).str.zfill(6)
    df = df.set_index("ticker")
    cache_set(key, df)
    return df


# ---------------------------------------------------------------- E7 listing metadata
def e7_listing_meta():
    """
    Reference metadata for currently-listed firms via FinanceDataReader ('KRX-DESC'),
    normalized to a 6-digit `ticker` index. Columns kept: Name, Market, Sector (KRX board
    section), Industry (KSIC text — the financials source, see CC_REPORT §a), SettleMonth
    (fiscal year-end, T3). Additive loader; no existing E signature changed. Point-in-time
    caveat: current listings only (delisted firms absent) — T unions with e6 for coverage.
    """
    key = "e7_listing_meta"
    c = cache_get(key)
    if c is not None:
        return c

    import FinanceDataReader as fdr
    raw = krx_call(fdr.StockListing, "KRX-DESC")
    df = raw.copy()
    df["ticker"] = df["Code"].astype(str).str.zfill(6)
    keep = [col for col in ["Name", "Market", "Sector", "Industry", "SettleMonth"]
            if col in df.columns]
    df = df.set_index("ticker")[keep]
    cache_set(key, df)
    return df


# ---------------------------------------------------------------- E8 adjusted monthly close
def e8_adjusted_monthly(ticker, start="19950101", end=None):
    """
    Per-ticker month-end adjusted close (pykrx freq='m', adjusted=True). Korean 수정주가
    adjusts splits + rights but NOT cash dividends, so this feeds a *price* monthly return
    (T6) that needs no manual split/rights correction. Adjusted close is per-ticker only
    (the cross-sectional E4 endpoint cannot adjust). Returns a `date`-indexed frame with a
    single `adj_close` column; empty frame if the ticker has no data in the window.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y%m%d")
    key = f"e8_adjm_{ticker}_{start}_{end}"
    c = cache_get(key)
    if c is not None:
        return c
    df = krx_call(stock.get_market_ohlcv, start, end, ticker, freq="m", adjusted=True)
    if df is None or len(df) == 0 or "종가" not in df.columns:
        out = pd.DataFrame({"adj_close": []})
        out.index.name = "date"
    else:
        out = df[["종가"]].rename(columns={"종가": "adj_close"})
        out.index.name = "date"
    cache_set(key, out)
    return out
