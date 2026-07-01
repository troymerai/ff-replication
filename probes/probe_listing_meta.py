"""
Recon probe for T-layer open items (§8 of the handover):
  (a) financial-sector identification  -> KRX-DESC `Sector` labels
  (d) non-December fiscal year-end mapping -> KRX-DESC `SettleMonth` labels
  (E6) delisting schema / common-stock filter -> KRX-DELISTING
  (b) dividend total-return path -> probe pykrx adjusted-close availability

Writes findings to scratch (UTF-8) so Korean labels survive the Windows console.
Run: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\probes\probe_listing_meta.py"
"""
import io
import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import FinanceDataReader as fdr

OUT = io.open("scratch_listing_meta.txt", "w", encoding="utf-8")
def w(*a):
    print(*a, file=OUT)

# ---- (a)(d) currently-listed metadata: Sector + SettleMonth ----
desc = fdr.StockListing("KRX-DESC")
w("=== KRX-DESC ===")
w("rows:", len(desc), "cols:", list(desc.columns))
mcol = "Market"
w("\nby market:")
w(desc[mcol].value_counts().to_string())

w("\n--- SettleMonth value counts (fiscal year-end) ---")
if "SettleMonth" in desc.columns:
    w(desc["SettleMonth"].value_counts(dropna=False).to_string())

w("\n--- Sector unique count ---")
if "Sector" in desc.columns:
    sec = desc["Sector"].dropna().unique()
    w("distinct Sector labels:", len(sec))
    w("\n--- Sector labels containing finance keywords ---")
    kw = ["금융", "은행", "보험", "증권", "신용", "저축", "캐피탈", "투자", "지주", "여신", "선물", "자산운용"]
    fin = [s for s in sec if any(k in str(s) for k in kw)]
    for s in sorted(fin):
        n = int((desc["Sector"] == s).sum())
        w(f"  {s!r}: {n}")
    w("\n--- ALL Sector labels (for manual audit) ---")
    for s in sorted(map(str, sec)):
        w("  ", s)

# ---- (E6) delisting ----
dl = fdr.StockListing("KRX-DELISTING")
w("\n\n=== KRX-DELISTING ===")
w("rows:", len(dl), "cols:", list(dl.columns))
w("\nSecuGroup value counts:")
w(dl["SecuGroup"].value_counts(dropna=False).to_string())
w("\nMarket value counts:")
w(dl["Market"].value_counts(dropna=False).to_string())
# common-stock KOSPI/KOSDAQ delistings with a parseable date
d = dl.copy()
d["DelistingDate"] = pd.to_datetime(d["DelistingDate"], errors="coerce")
common = d[(d["SecuGroup"] == "주권") & (d["Market"].isin(["KOSPI", "KOSDAQ"]))]
w("\ncommon-stock (주권) KOSPI/KOSDAQ delistings:", len(common))
w("  with valid DelistingDate:", int(common["DelistingDate"].notna().sum()))
w("  date range:", common["DelistingDate"].min(), "~", common["DelistingDate"].max())
w("\nReason value counts (top 15) for common-stock delistings:")
w(common["Reason"].value_counts(dropna=False).head(15).to_string())
w("\nsample rows around 2010:")
s2010 = common[(common["DelistingDate"] >= "2009-07-01") & (common["DelistingDate"] < "2010-07-01")]
w("  delistings in 2009-07..2010-06:", len(s2010))
w(s2010[["Symbol", "Name", "Market", "DelistingDate", "Reason"]].head(10).to_string(index=False))

# ---- (b) dividend / adjusted price probe ----
w("\n\n=== (b) dividend total-return path probe ===")
from pykrx import stock
import inspect
try:
    sig = inspect.signature(stock.get_market_ohlcv)
    w("get_market_ohlcv signature:", str(sig))
except Exception as e:
    w("signature introspection failed:", e)
try:
    sig2 = inspect.signature(stock.get_market_ohlcv_by_date)
    w("get_market_ohlcv_by_date signature:", str(sig2))
except Exception as e:
    w("by_date signature failed:", e)
# does a single-ticker time series support adjusted=?
try:
    raw = stock.get_market_ohlcv("20100601", "20100701", "005930", adjusted=False)
    adj = stock.get_market_ohlcv("20100601", "20100701", "005930", adjusted=True)
    w("\n005930 raw vs adjusted close (2010-06), last 3 rows:")
    w("raw:\n" + raw["종가"].tail(3).to_string())
    w("adj:\n" + adj["종가"].tail(3).to_string())
    w("adjusted differs from raw:", not raw["종가"].equals(adj["종가"]))
except Exception as e:
    w("adjusted time-series probe failed:", e)

OUT.close()
print("wrote scratch_listing_meta.txt")
