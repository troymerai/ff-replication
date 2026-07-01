"""Verify the split-adjustment formula direction on Samsung's 2018-05 50:1 split,
using RAW cross-sections (E4 close + E3 shares) — the same data available back to 2002.
Ground truth adjusted return from the adjusted feed = -0.0434."""
import io
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ff_kr_extract as E

L = []
px0 = E.e4_prices("20180430", "KOSPI")["frame"].loc["005930", "종가"]
px1 = E.e4_prices("20180531", "KOSPI")["frame"].loc["005930", "종가"]
sh0 = E.e3_marketcap("20180430", "KOSPI")["frame"].loc["005930", "상장주식수"]
sh1 = E.e3_marketcap("20180531", "KOSPI")["frame"].loc["005930", "상장주식수"]
mc0 = E.e3_marketcap("20180430", "KOSPI")["frame"].loc["005930", "시가총액"]
mc1 = E.e3_marketcap("20180531", "KOSPI")["frame"].loc["005930", "시가총액"]
L.append(f"2018-04: close={int(px0):,} shares={int(sh0):,} cap={int(mc0):,}")
L.append(f"2018-05: close={int(px1):,} shares={int(sh1):,} cap={int(mc1):,}")
k = sh1 / sh0
pr = px1 / px0
L.append(f"k = shares_t/shares_(t-1) = {k:.4f}   price_ratio = {pr:.5f}   1/k = {1/k:.5f}")
L.append(f"RAW return (fake)      = {pr - 1:.4f}")
L.append(f"ADJ return = pr*k - 1  = {pr*k - 1:.4f}   <- must match adjusted-feed truth -0.0434")
L.append(f"CAP return = mc1/mc0-1 = {mc1/mc0 - 1:.4f}")
(ROOT / "scratch_splitchk.txt").write_text("\n".join(L), encoding="utf-8")
print("OK")
