"""
(1) Confirm pykrx adjusted vs raw close across a known split (005930, 50:1 May 2018)
    -> if adjusted is continuous while raw shows a ~50x drop, T6 uses adjusted close and
       needs NO manual split/rights correction. Also confirm adjusted is per-ticker only.
(2) Dump full `기타 금융업` names so the financial-holding-company allowlist (FSS list,
    keep industrial holdcos) can be built precisely.
"""
import io
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from pykrx import stock
import FinanceDataReader as fdr

OUT = io.open("scratch_adj_holdco.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

# ---- (1) adjusted vs raw across the 2018-05 Samsung 50:1 split ----
w("=== (1) 005930 monthly close 2018-03..2018-08: raw vs adjusted ===")
raw = stock.get_market_ohlcv("20180301", "20180831", "005930", freq="m", adjusted=False)
adj = stock.get_market_ohlcv("20180301", "20180831", "005930", freq="m", adjusted=True)
cmp = pd.DataFrame({"raw_close": raw["종가"], "adj_close": adj["종가"]})
cmp["raw_ret"] = cmp["raw_close"].pct_change()
cmp["adj_ret"] = cmp["adj_close"].pct_change()
w(cmp.to_string())
w("\n-> raw shows a fake ~-98% at the split month; adj_ret should be ~real (single-digit %).")

# is adjusted available cross-sectionally? (single-date market call)
w("\n=== adjusted cross-section? ===")
try:
    xs = stock.get_market_ohlcv("20180531", market="KOSPI", adjusted=True)
    w("cross-section adjusted=True returned rows:", len(xs), "(param accepted)")
except TypeError as e:
    w("cross-section does NOT accept adjusted:", e)

# ---- (2) full 기타 금융업 names ----
w("\n\n=== (2) 기타 금융업 full name list (build financial-holdco allowlist) ===")
desc = fdr.StockListing("KRX-DESC")
etc = desc[desc["Industry"] == "기타 금융업"][["Code", "Name", "Market"]].sort_values("Name")
for _, r in etc.iterrows():
    w(f"  {r['Code']}  {r['Name']}  ({r['Market']})")
w("\ncount:", len(etc))

OUT.close()
print("done")
