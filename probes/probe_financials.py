"""
Follow-up recon: how to identify financial firms (§8-a).
KRX-DESC.Sector was the KRX board section (소속부), not industry. Try:
  - StockListing('KRX')  -> may carry a real industry/sector column
  - KRX-DESC.Industry    -> business-description text (keyword match)
  - pykrx sector/index membership as a cross-check
"""
import io
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import FinanceDataReader as fdr

OUT = io.open("scratch_financials.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

FIN_KW = ["은행", "보험", "증권", "금융", "신용", "저축", "캐피탈", "여신",
          "선물", "자산운용", "투자자문", "지주", "리스", "카드", "상호저축"]

# ---- StockListing('KRX') ----
try:
    k = fdr.StockListing("KRX")
    w("=== StockListing('KRX') ===")
    w("rows:", len(k), "cols:", list(k.columns))
    for cand in ["Sector", "Industry", "IndustryCode", "SectorCode"]:
        if cand in k.columns:
            vals = k[cand].dropna().unique()
            w(f"\n{cand}: {len(vals)} distinct")
            fin = sorted({v for v in vals if any(kw in str(v) for kw in FIN_KW)})
            w(f"  finance-keyword labels ({len(fin)}):")
            for v in fin:
                w("    ", v, "->", int((k[cand] == v).sum()))
except Exception as e:
    w("StockListing('KRX') failed:", repr(e))

# ---- KRX-DESC.Industry ----
try:
    desc = fdr.StockListing("KRX-DESC")
    w("\n\n=== KRX-DESC.Industry keyword scan ===")
    if "Industry" in desc.columns:
        ind = desc["Industry"].fillna("")
        mask = ind.apply(lambda s: any(kw in s for kw in FIN_KW))
        w("rows matching finance keywords:", int(mask.sum()), "/", len(desc))
        w("\ndistinct matching Industry labels:")
        for v, n in desc.loc[mask, "Industry"].value_counts().items():
            w(f"  {v!r}: {n}")
        w("\nby market among matches:")
        w(desc.loc[mask, "Market"].value_counts().to_string())
except Exception as e:
    w("KRX-DESC.Industry scan failed:", repr(e))

# ---- pykrx: does it expose sector/industry classification? ----
try:
    from pykrx import stock
    w("\n\n=== pykrx sector/index cross-check ===")
    # KRX industry indices membership (e.g., 금융업 index) as a fallback identifier
    idx = stock.get_index_ticker_list("20100630", market="KOSPI")
    names = {t: stock.get_index_ticker_name(t) for t in idx}
    fin_idx = {t: n for t, n in names.items() if any(kw in str(n) for kw in FIN_KW + ["금융업"])}
    w("KOSPI index tickers with finance-related names:")
    for t, n in fin_idx.items():
        w("  ", t, n)
except Exception as e:
    w("pykrx sector cross-check failed:", repr(e))

OUT.close()
print("wrote scratch_financials.txt")
