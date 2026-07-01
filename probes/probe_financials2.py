"""Refine the financial-industry allowlist: inspect the ambiguous buckets."""
import io
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import FinanceDataReader as fdr

OUT = io.open("scratch_financials2.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

desc = fdr.StockListing("KRX-DESC")
FIN_LABELS = ['기타 금융업', '금융 지원 서비스업', '보험업', '은행 및 저축기관',
              '보험 및 연금관련 서비스업', '재 보험업']
for lab in FIN_LABELS:
    sub = desc[desc["Industry"] == lab]
    spac = sub["Name"].str.contains("스팩|기업인수목적", na=False)
    holdco = sub["Name"].str.contains("지주|홀딩스|Holdings", case=False, na=False)
    w(f"=== {lab}  (n={len(sub)}) ===")
    w(f"  SPAC-named: {int(spac.sum())} | holdco-named: {int(holdco.sum())}")
    w("  sample non-SPAC names: " + ", ".join(sub.loc[~spac, "Name"].head(12).tolist()))
    w("")

# delisting frame Industry labels (for point-in-time union)
dl = fdr.StockListing("KRX-DELISTING")
common = dl[(dl["SecuGroup"] == "주권") & (dl["Market"].isin(["KOSPI", "KOSDAQ"]))]
w("\n=== KRX-DELISTING.Industry finance-related labels (common stock) ===")
FIN_KW = ["금융", "은행", "보험", "증권", "저축", "캐피탈", "여신", "선물", "자산운용"]
ind = common["Industry"].fillna("")
mask = ind.apply(lambda s: any(k in s for k in FIN_KW))
w("matching rows:", int(mask.sum()), "/", len(common))
w(common.loc[mask, "Industry"].value_counts().to_string())
w("\nIndustry column non-null in delisting frame:", int(common["Industry"].notna().sum()), "/", len(common))

OUT.close()
print("done")
