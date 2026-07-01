# probe_kosdaq_byticker_2003.py
# ─────────────────────────────────────────────────────────────────────────
# KOSDAQ by-ticker PBR depth — can 2003-2005 be recovered by assembly?
#
# Background: in the probe_pbr_xsec_bottom re-scan, the KOSDAQ cross-section is
#       entirely empty for 2003-2005 (Dec) and 2003-2005 (Jun) across the whole
#       15-day window (ok=False). For KOSPI, even where the cross-section was
#       blank, by-ticker data existed back to 2002. If KOSDAQ likewise exists
#       per ticker, only the early window needs by-ticker assembly, and
#       KOSPI+KOSDAQ can run from 2003 with no seam. If not, the common start is
#       fixed at 2006.
#
# Run: python probes/probe_kosdaq_byticker_2003.py
# ─────────────────────────────────────────────────────────────────────────

# %% [0] login (.env first) + imports
import os, warnings, random
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [1] sample from the KOSDAQ universe as of 2003
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
univ = stock.get_market_ticker_list(td_base, market="KOSDAQ")   # KOSDAQ listed as of 2003
random.seed(42)
probe_tickers = random.sample(univ, min(40, len(univ)))
print(f"2003 KOSDAQ universe: {len(univ)} stocks · probe sample: {len(probe_tickers)}")

# %% [2] by-ticker PBR validity — early window (2003/2004/2005, June & December each)
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

print(f"\nby-ticker responding stocks: {tot}/{len(probe_tickers)}")
for d in CHECK:
    share = f"{hit[d]}/{tot}" if tot else "0/0"
    print(f"  {d} PBR>0: {share}")

# %% [3] verdict
print("\n" + "=" * 60)
print("verdict")
print("=" * 60)
if tot == 0:
    print("by-ticker responses: 0 — per-ticker endpoints cannot supply early KOSDAQ either. Common start fixed at 2006.")
else:
    early = sum(hit[d] for d in ["2003-06", "2003-12", "2004-12"]) / (3 * tot)
    print(f"2003-2004 average validity rate ≈ {early:.0%}")
    print("  high (e.g. >= 60%) → KOSDAQ 2003-2005 can be assembled by ticker → KOSPI+KOSDAQ from 2003 with no seam.")
    print("  low or 0           → early KOSDAQ genuinely absent → common start 2006 (primary) + KOSPI-only 2003- (secondary).")
print("\nNote: even if recoverable, the assembly cost (per-ticker pulls) and consistency (cross-section vs by-ticker values) are confirmed in the extraction stage.")
