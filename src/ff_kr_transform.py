"""
Korean FF3 data transformation — T (Transform) layer.

Consumes the E-layer raw cross-sections and produces a point-in-time clean panel
(the input to factor construction, 5b). Implements T1-T10 of the handover:

    T1  ticker normalization + universe filter (KOSPI/KOSDAQ common, ex-financials, ex-SPAC)
    T2  point-in-time universe (no lookahead)
    T3  accounting lag (fiscal year-end via SettleMonth)
    T4  B/M = 1/PBR from the prior-December KRX cross-section (PBR>0; neg-BE dropped, D6)
    T5  two distinct ME points: me_6 (size sort) and me_12 (B/M denominator)
    T6  monthly returns from per-ticker ADJUSTED close (splits/rights handled; div omitted)
    T7  delisting: hold to last trading day, drop after (no carry-forward), log
    T8  trading halts handled for return calc only (no liquidity screen)
    T9  breakpoints (size median / B/M 30-70; breakpoint universe parameterized)
    T10 portfolio assignment: 5x5 (size x B/M) test assets + 2x3 factor grid

Methodology decisions (D1-D11) and the resolution of the handover's open items (financials
source, dividend handling, adjusted close) are documented in CC_REPORT.md.

Financials are identified from KRX-DESC.Industry (KSIC) unioned with the KRX-DELISTING
Industry labels; financial holding companies are excluded while industrial holdcos are
kept (flagged for D11). See CC_REPORT.md §Stage 2.

New code is English per repo convention; the pykrx/KRX Korean column and label strings
(시가총액, 상장주식수, 종가, 기타 금융업, …) are functional keys and kept verbatim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow "import ff_kr_extract"
import ff_kr_extract as E

# ---------------------------------------------------------------- config / parameters
MARKETS = E.MARKETS
KOSDAQ_BM_ONSET = E.KOSDAQ_BM_ONSET

# --- financial-firm identification (CC_REPORT §Stage 2) -------------------------------
# KSIC finance-division labels excluded wholesale.
FIN_INDUSTRY_WHOLESALE = {
    "보험업", "은행 및 저축기관", "재 보험업", "보험 및 연금관련 서비스업",
}
# 금융 지원 서비스업 = securities firms (non-SPAC) + SPACs; the securities are financial.
FIN_SUPPORT_LABEL = "금융 지원 서비스업"
# 기타 금융업 (KSIC 64.9) mixes financial holdcos with industrial holdcos; exclude only
# the financial members: FSS-designated financial holding companies + finance-name firms.
ETC_FIN_LABEL = "기타 금융업"
FIN_HOLDCO_TICKERS = {          # FSS-designated financial holding companies (listed)
    "138930",  # BNK금융지주
    "175330",  # JB금융지주
    "105560",  # KB금융
    "139130",  # iM금융지주 (former DGB)
    "138040",  # 메리츠금융지주
    "055550",  # 신한지주
    "316140",  # 우리금융지주
    "086790",  # 하나금융지주
    "071050",  # 한국금융지주
}
ETC_FIN_NAME_KW = ("증권", "투자", "인베스트", "캐피탈", "카드", "벤처", "파트너스", "금융")
# KRX-DELISTING.Industry uses short labels; these are financial.
DELIST_FIN_INDUSTRY_KW = ("금융", "은행", "보험", "증권")

SPAC_NAME_KW = ("스팩", "기업인수목적")
HOLDCO_NAME_KW = ("지주", "홀딩스", "Holdings", "HOLDINGS")

# --- sort parameters -----------------------------------------------------------------
BREAKPOINT_UNIVERSES = ("kospi_only", "whole_market")
DEFAULT_BREAKPOINT_UNIVERSE = "kospi_only"   # FF NYSE-only analogue (D8, not final)
MAX_MONTHLY_RETURN = 3.0                      # |monthly| > 300% flagged as implausible (§10)


# ---------------------------------------------------------------- T1 helpers: classification
def _norm(ticker) -> str:
    """6-digit zero-padded ticker string."""
    return str(ticker).zfill(6)


def _is_spac(name: str) -> bool:
    return any(kw in name for kw in SPAC_NAME_KW)


def _is_holdco(name: str) -> bool:
    return any(kw in name for kw in HOLDCO_NAME_KW)


def _is_financial(ticker: str, industry: str, name: str) -> bool:
    """Financial-firm test on current KRX-DESC metadata (CC_REPORT §Stage 2)."""
    industry = industry or ""
    name = name or ""
    if industry in FIN_INDUSTRY_WHOLESALE:
        return True
    if industry == FIN_SUPPORT_LABEL and not _is_spac(name):
        return True                                  # securities firm
    if industry == ETC_FIN_LABEL:
        if ticker in FIN_HOLDCO_TICKERS:
            return True                              # financial holding company
        if any(kw in name for kw in ETC_FIN_NAME_KW):
            return True                              # securities / investment / capital / card
    return False


def reference_maps():
    """
    Build ticker -> (name, industry) maps from current listings (E7) unioned with the
    full delisting history (E6), so point-in-time universes of any date can be classified
    without per-ticker name calls. Current metadata takes precedence; delisted names fill
    gaps. Returns (meta_df, delist_df, name_map, financial_set).
    """
    meta = E.e7_listing_meta()
    dl = E.e6_delisting()

    name_map: dict[str, str] = {}
    for tk, row in meta.iterrows():
        name_map[tk] = str(row.get("Name") or "")
    for tk, row in dl.iterrows():
        name_map.setdefault(_norm(tk), str(row.get("Name") or ""))

    financial: set[str] = set()
    for tk, row in meta.iterrows():
        if _is_financial(_norm(tk), str(row.get("Industry") or ""), str(row.get("Name") or "")):
            financial.add(_norm(tk))
    for tk, row in dl.iterrows():
        ind = str(row.get("Industry") or "")
        if any(kw in ind for kw in DELIST_FIN_INDUSTRY_KW):
            financial.add(_norm(tk))
    return meta, dl, name_map, financial


# ---------------------------------------------------------------- T1/T2 universe
def build_universe(sort_date: str, market: str, name_map: dict, financial: set) -> dict:
    """
    Point-in-time common-stock universe for one market at the sort date, after filters.
    Returns dict(tickers, dropped) where dropped breaks the exclusions down for audit.

    Filters (T1): KONEX already excluded by querying only KOSPI/KOSDAQ; preferred shares
    dropped by the KRX common-stock code convention (6th digit '0'); financials excluded;
    SPACs excluded. No liquidity screen (D5). No lookahead (T2): the E1 list is the set
    of tickers actually listed on the sort date.
    """
    raw = [_norm(t) for t in E.e1_universe(sort_date, market)]
    kept, dropped = [], {"preferred": [], "financial": [], "spac": []}
    for tk in raw:
        if tk[-1] != "0":                     # preferred / non-common share class
            dropped["preferred"].append(tk)
            continue
        if tk in financial:
            dropped["financial"].append(tk)
            continue
        if _is_spac(name_map.get(tk, "")):
            dropped["spac"].append(tk)
            continue
        kept.append(tk)
    return {"tickers": kept, "dropped": dropped}


# ---------------------------------------------------------------- T3 accounting lag
def prior_december(sort_date: str) -> str:
    """Prior-December cross-section date (t-1 Dec 31) for a June-t sort. Blank-date
    walk-back is handled inside the E loaders (Dec 31 is often non-trading)."""
    d = pd.Timestamp(sort_date)
    return f"{d.year - 1}1231"


def settle_month_map(meta: pd.DataFrame) -> dict:
    """ticker -> fiscal settle month (int 1..12); missing/unknown default to 12 (Dec).
    Non-December firms (< 2% of listings) carry a known v0.1 timing caveat: their book is
    taken from the KRX December PBR cross-section, not their own fiscal close (CC_REPORT §d)."""
    out = {}
    if "SettleMonth" not in meta.columns:
        return out
    for tk, val in meta["SettleMonth"].items():
        s = str(val)
        m = 12
        for k in range(1, 13):
            if f"{k:02d}월" == s or f"{k}월" == s:
                m = k
                break
        out[_norm(tk)] = m
    return out


# ---------------------------------------------------------------- T4/T5 sort cross-section
def build_sort_cross_section(sort_date: str,
                             markets=MARKETS,
                             refs=None) -> dict:
    """
    Assemble the point-in-time sort-date cross-section for T4 (B/M) and T5 (two ME points).

    Per market and joined on the ticker index:
      B/M (T4)       : 1 / PBR, PBR from prior-December e2_fundamentals (PBR>0 -> drops
                       negative book equity, D6)
      me_6 (T5 size) : 시가총액 from June-t e3_marketcap
      me_12 (T5 B/M) : 시가총액 from prior-December e3_marketcap (carried for the two-ME
                       integrity check; the v0.1 B/M ratio itself is 1/PBR at Dec t-1)

    Returns dict(frame, meta) where frame is indexed by ticker with the sort attributes and
    audit columns (used_fund_date, walked_fund, market, name, is_holdco_dual), and meta
    records per-market join counts and skips.
    """
    if refs is None:
        refs = reference_maps()
    meta_desc, dl, name_map, financial = refs
    smap = settle_month_map(meta_desc)
    dec_date = prior_december(sort_date)

    frames, per_market = [], {}
    for market in markets:
        uni = build_universe(sort_date, market, name_map, financial)
        tickers = uni["tickers"]

        fund = E.e2_fundamentals(dec_date, market)         # B/M from prior Dec
        cap6 = E.e3_marketcap(sort_date, market)           # size (June ME)
        cap12 = E.e3_marketcap(dec_date, market)           # Dec ME (B/M denominator)

        rec = {"universe": len(tickers), "dropped": {k: len(v) for k, v in uni["dropped"].items()}}
        if fund.get("skip"):
            rec["skip"] = fund["skip"]
        if not (fund["ok"] and cap6["ok"] and cap12["ok"]):
            rec["joined"] = 0
            rec["loader_ok"] = {"fund": fund["ok"], "cap6": cap6["ok"], "cap12": cap12["ok"]}
            per_market[market] = rec
            continue

        f = fund["frame"][["BPS", "PBR"]].copy()
        f.index = [_norm(t) for t in f.index]
        c6 = cap6["frame"][["시가총액"]].rename(columns={"시가총액": "me_6"})
        c6.index = [_norm(t) for t in c6.index]
        c12 = cap12["frame"][["시가총액"]].rename(columns={"시가총액": "me_12"})
        c12.index = [_norm(t) for t in c12.index]

        uni_idx = pd.Index(tickers, name="ticker")
        df = (f.reindex(uni_idx)
                .join(c6, how="left")
                .join(c12, how="left"))
        df = df[(df["PBR"] > 0) & (df["me_6"] > 0)]        # D6 neg-BE drop + valid size
        df["bm"] = 1.0 / df["PBR"]
        df["market"] = market
        df["name"] = [name_map.get(t, "") for t in df.index]
        df["is_holdco_dual"] = [(_is_holdco(name_map.get(t, ""))) for t in df.index]
        df["settle_month"] = [smap.get(t, 12) for t in df.index]
        df["used_fund_date"] = fund.get("used")
        df["walked_fund"] = fund.get("walked")
        df["used_cap6_date"] = cap6.get("used")

        rec["joined"] = len(df)
        rec["bm_median"] = float(df["bm"].median())
        per_market[market] = rec
        frames.append(df)

    panel = pd.concat(frames) if frames else pd.DataFrame()
    return {"frame": panel, "meta": {"sort_date": sort_date, "dec_date": dec_date,
                                     "per_market": per_market}}


# ---------------------------------------------------------------- T9 breakpoints
def _breakpoints(df: pd.DataFrame, breakpoint_universe: str) -> dict:
    """Size median + B/M 30/70 (and size/B/M quintiles) computed on the breakpoint
    universe and applied to all stocks (FF NYSE-only analogue). D8 parameterized."""
    if breakpoint_universe == "kospi_only":
        bp = df[df["market"] == "KOSPI"]
        if len(bp) == 0:                      # e.g. KOSPI-only-skip periods -> fall back
            bp = df
    else:
        bp = df
    return {
        "size_median": bp["me_6"].median(),
        "size_q": bp["me_6"].quantile([0.2, 0.4, 0.6, 0.8]).tolist(),
        "bm_30": bp["bm"].quantile(0.30),
        "bm_70": bp["bm"].quantile(0.70),
        "bm_q": bp["bm"].quantile([0.2, 0.4, 0.6, 0.8]).tolist(),
        "n_breakpoint": len(bp),
    }


def _assign_quintile(x, cuts):
    return int(np.searchsorted(cuts, x, side="right")) + 1     # 1..5


# ---------------------------------------------------------------- T10 portfolio assignment
def assign_portfolios(df: pd.DataFrame,
                      breakpoint_universe: str = DEFAULT_BREAKPOINT_UNIVERSE) -> dict:
    """
    Assign each stock to the 2x3 factor grid and the 5x5 test grid using breakpoints from
    the chosen universe. Adds columns: size2 (S/B), bm3 (L/M/H), size5, bm5, port_2x3,
    port_5x5, breakpoint_flag. Returns dict(frame, breakpoints, cell_counts).
    """
    df = df.copy()
    bp = _breakpoints(df, breakpoint_universe)

    df["size2"] = np.where(df["me_6"] <= bp["size_median"], "S", "B")
    df["bm3"] = np.where(df["bm"] <= bp["bm_30"], "L",
                         np.where(df["bm"] >= bp["bm_70"], "H", "M"))
    df["size5"] = df["me_6"].apply(lambda x: _assign_quintile(x, bp["size_q"]))
    df["bm5"] = df["bm"].apply(lambda x: _assign_quintile(x, bp["bm_q"]))
    df["port_2x3"] = df["size2"] + "/" + df["bm3"]
    df["port_5x5"] = ("S" + df["size5"].astype(str) + "B" + df["bm5"].astype(str))
    df["breakpoint_flag"] = breakpoint_universe

    cells = {
        "2x3": df["port_2x3"].value_counts().sort_index().to_dict(),
        "5x5_n": int(df["port_5x5"].nunique()),
    }
    return {"frame": df, "breakpoints": bp, "cell_counts": cells}


# ---------------------------------------------------------------- T6/T7/T8 returns
SHARE_EPS = 0.03         # |k-1| below this = no real corporate action (minor drift -> price return)
SPLIT_TOL = 0.35         # |cap_ratio-1| below this with a share change = cap-preserving split/bonus


def month_end_dates(start: str, end: str) -> list[str]:
    """Calendar month-end dates (yyyymmdd) inclusive of both ends; the E loaders walk each
    back to the nearest trading day. `start` is the base month-end (the sort date), so the
    first return is start->next month-end."""
    rng = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")
    ds = {d.strftime("%Y%m%d") for d in rng}
    ds.add(pd.Timestamp(start).strftime("%Y%m%d"))       # include the base even if not a ME
    ds.add(pd.Timestamp(end).strftime("%Y%m%d"))
    return sorted(ds)


def _cap_shares_panel(dates, markets) -> dict:
    """Wide {date -> DataFrame(cap, shares) indexed by ticker} from E3 cross-sections
    (시가총액 + 상장주식수), each date resolved (walk-back) inside E3. Full history 2002+."""
    panel = {}
    for ds in dates:
        parts = []
        for market in markets:
            r = E.e3_marketcap(ds, market)
            if not r["ok"]:
                continue
            f = r["frame"][["시가총액", "상장주식수"]].rename(
                columns={"시가총액": "cap", "상장주식수": "shares"})
            f.index = [_norm(t) for t in f.index]
            parts.append(f)
        if parts:
            panel[ds] = pd.concat(parts)
    return panel


def _classify_return(cap0, cap1, sh0, sh1):
    """
    Monthly return with split handling (CC_REPORT §Stage 4). Returns (ret, flag).

    Shares ~constant  -> price return (= cap return; no corporate action).
    Material share change -> MARKET-CAP return `cap_ratio - 1`. This is robust to splits /
    reverse-merges / bonus issues (cap-preserving, per-share value continuous) and, crucially,
    never uses the split-contaminated `price_ratio` (e.g. a 50:1 reverse merge makes
    price_ratio ~50x, which as a "return" is nonsense). For a genuine capital raise / CB /
    M&A the cap return overstates the holder return by the new-capital share — a bounded v0.1
    approximation, sub-flagged `share_change_capex` and logged; splits/bonus are `split_bonus`.
    """
    if cap0 <= 0 or cap1 <= 0 or sh0 <= 0 or sh1 <= 0:
        return None, "invalid"
    k = sh1 / sh0
    cap_ratio = cap1 / cap0
    price_ratio = cap_ratio / k
    if abs(k - 1.0) < SHARE_EPS:
        return price_ratio - 1.0, "none"
    flag = "split_bonus" if abs(cap_ratio - 1.0) < SPLIT_TOL else "share_change_capex"
    return cap_ratio - 1.0, flag


def build_returns(tickers, hold_start: str, hold_end: str,
                  markets=MARKETS, delist: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Monthly returns over [hold_start, hold_end] built by stitching consecutive month-end E3
    cross-sections (시가총액 + 상장주식수). price = cap/shares; splits/bonuses are removed via
    market-cap continuity, capital raises are not (CC_REPORT §Stage 4). Full history 2002+,
    no per-ticker calls. Dividends omitted (v0.1, price return, logged).

    T7 delisting: a ticker absent from a later cross-section has no return that month, so its
    series ends at its last trading month (no carry-forward); E6 delisting date sets is_delisted
    and drops any month strictly after it. T8 halts: a one-month gap yields no return for the
    gap month (not a liquidity screen). Implausible months (|ret| > MAX_MONTHLY_RETURN) flagged.

    Returns long frame: [date, ticker, ret_m, adj_flag, is_delisted, ret_flag].
    """
    tickers = {_norm(t) for t in tickers}
    dates = month_end_dates(hold_start, hold_end)
    panel = _cap_shares_panel(dates, markets)
    ordered = [d for d in dates if d in panel]

    delist_dates = {}
    if delist is not None:
        for tk, row in delist.iterrows():
            dd = row.get("DelistingDate")
            if pd.notna(dd):
                delist_dates[_norm(tk)] = pd.Timestamp(dd)

    rows, dropped = [], []
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        a, b = panel[prev], panel[cur]
        common = (set(a.index) & set(b.index)) & tickers
        cur_ts = pd.Timestamp(cur)                       # report the return at the end month
        for tk in common:
            dd = delist_dates.get(tk)
            if dd is not None and cur_ts > dd:
                continue                                 # T7 no carry-forward past delisting
            ret, flag = _classify_return(a.at[tk, "cap"], b.at[tk, "cap"],
                                         a.at[tk, "shares"], b.at[tk, "shares"])
            if ret is None:
                continue
            rec = {
                "date": cur_ts, "ticker": tk, "ret_m": float(ret), "adj_flag": flag,
                "is_delisted": bool(dd is not None and dd <= pd.Timestamp(hold_end)),
            }
            if abs(ret) > MAX_MONTHLY_RETURN:            # §10: implausible dropped, logged
                dropped.append(rec)
            else:
                rows.append(rec)
    out = pd.DataFrame(rows)
    out.attrs["dropped_implausible"] = pd.DataFrame(dropped)
    return out


# ---------------------------------------------------------------- integrity checks (§10)
def integrity_checks(sort_df: pd.DataFrame) -> dict:
    """6a integrity pre-check on the sort cross-section. Returns dict(passed, details)."""
    d = {}
    d["missing_ret_fields"] = int(sort_df[["me_6", "bm"]].isna().sum().sum())
    d["negative_me"] = int((sort_df["me_6"] <= 0).sum())
    d["negative_or_zero_bm"] = int((sort_df["bm"] <= 0).sum())
    # T5 two-ME consistency: me_6 (June) and me_12 (Dec) must both exist and generally
    # differ (same value across two dates would signal a join/date bug).
    both = sort_df.dropna(subset=["me_6", "me_12"])
    d["two_me_present"] = int(len(both))
    d["two_me_identical"] = int((both["me_6"] == both["me_12"]).sum())
    d["two_me_identical_frac"] = (float((both["me_6"] == both["me_12"]).mean())
                                  if len(both) else float("nan"))
    d["passed"] = (d["missing_ret_fields"] == 0 and d["negative_me"] == 0
                   and d["negative_or_zero_bm"] == 0
                   and (d["two_me_identical_frac"] < 0.5 if len(both) else False))
    return d
