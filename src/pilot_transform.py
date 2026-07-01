"""
T+L pilot — one rebalance (2010-06 sort, prior Dec 2009), driven through the L orchestrator.
Validates against handover §6 anchors and §10 integrity, exercises the stitched return panel,
and writes the §9 monthly-long panel to results/. Writes a report block (UTF-8) and prints a
concise status. Cross-sectional validation is the primary gate; the return pull is the full
pilot universe over the 12-month holding window.

Run: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\src\pilot_transform.py"
"""
import io
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import ff_kr_extract as E
import ff_kr_transform as T
import ff_kr_load as L

SORT_DATE = "20100630"
OUT = io.open(ROOT / "scratch_pilot.txt", "w", encoding="utf-8")
def w(*a): print(*a, file=OUT)

w("# T+L pilot — sort", SORT_DATE, "/ prior Dec", T.prior_december(SORT_DATE))

# ---- §6 raw-join anchor (E level, no T filters): expect KOSPI 673, KOSDAQ 960 ----
w("\n## §6 raw-join anchor (E-level inner join, no T filters)")
dec = T.prior_december(SORT_DATE)
for market in E.MARKETS:
    fund = E.e2_fundamentals(dec, market); cap = E.e3_marketcap(SORT_DATE, market)
    px = E.e4_prices(SORT_DATE, market)
    if not (fund["ok"] and cap["ok"] and px["ok"]):
        w(f"  {market}: loader skip ({fund.get('skip')})"); continue
    f = fund["frame"][["BPS", "PBR"]]; c = cap["frame"][["시가총액"]]
    p = px["frame"][["종가"]] if "종가" in px["frame"].columns else px["frame"].iloc[:, [3]]
    j = f.join(c, how="inner").join(p, how="inner")
    j = j[(j["PBR"] > 0) & (j["시가총액"] > 0)]
    w(f"  {market}: raw join {len(j)}  | B/M median {1.0/j['PBR'].median():.3f}")

# ---- full T+L pipeline via the orchestrator (single pass) ----
refs = T.reference_maps()
r = L.run_rebalance(SORT_DATE, refs=refs)
sort, returns, panel, logs = r["sort"], r["returns"], r["panel"], r["logs"]

w("\n## T pipeline (filtered universe)")
for market, rec in logs["meta"]["per_market"].items():
    w(f"  {market}: universe(after T1) {rec['universe']} | dropped {rec.get('dropped')}"
      f" | joined {rec.get('joined')} | B/M median {rec.get('bm_median')}")
w(f"\n  sort cross-section rows: {len(sort)}  markets: {sort['market'].value_counts().to_dict()}")
w(f"  holdco-dual flagged: {int(sort['is_holdco_dual'].sum())}")
w(f"  walk-back (fund) used date: {sort['used_fund_date'].unique().tolist()}")

# 기타 금융업 audit
meta_desc = refs[0]
etc = meta_desc[meta_desc["Industry"] == "기타 금융업"]
excl = [T._norm(t) for t in etc.index if T._norm(t) in refs[3]]
w(f"\n## 기타 금융업 audit: excluded {len(excl)} / kept {len(etc) - len(excl)} (of {len(etc)})")
w("  excluded: " + ", ".join(sorted(str(meta_desc.loc[t, 'Name']) for t in excl if t in meta_desc.index)))

# breakpoints / portfolios
bp = r["breakpoints"]
w("\n## T9/T10 portfolios (breakpoint universe = kospi_only)")
w(f"  size median (KOSPI) {bp['size_median']:,.0f} | B/M 30/70 {bp['bm_30']:.3f}/{bp['bm_70']:.3f}"
  f" | n_breakpoint {bp['n_breakpoint']}")
w(f"  2x3 cells: {logs['cell_counts']['2x3']}")
w("  5x5 (size5 x bm5):")
w(pd.crosstab(sort["size5"], sort["bm5"]).to_string())

# integrity
w("\n## §10 integrity (sort cross-section)")
for k, v in logs["integrity"].items():
    w(f"  {k}: {v}")

# returns
w("\n## T6/T7/T8 return panel (stitched E3 cross-sections, hold 2010-07..2011-06)")
dropped = logs["dropped_implausible"]
w(f"  return rows: {len(returns)}  tickers: {returns['ticker'].nunique()}"
  f"  months: {returns['date'].nunique()}")
w(f"  monthly ret mean {returns['ret_m'].mean():.4f} median {returns['ret_m'].median():.4f}"
  f" min {returns['ret_m'].min():.4f} max {returns['ret_m'].max():.4f}")
w(f"  adj_flag counts: {logs['adj_flag_counts']}")
w(f"  implausible (>|300%|) DROPPED: {len(dropped)} -> panel max |ret| {returns['ret_m'].abs().max():.3f}")
w(f"  delisted-in-window rows: {int(returns['is_delisted'].sum())}")

# ---- L layer: assemble + persist ----
w("\n## L panel (§9 monthly-long)")
w(f"  panel rows: {len(panel)}  columns: {list(panel.columns)}")
miss = panel[["ret_m", "mktcap_6", "bm"]].isna().sum().to_dict()
w(f"  missing in required fields (ret_m/mktcap_6/bm): {miss}")
paths = L.write_panel(panel, "panel_pilot_2010", formats=("csv", "sqlite", "parquet"))
w(f"  written: {paths}")

# final completion-criteria (§10) roll-up
w("\n## §10 completion roll-up")
w(f"  [{'OK' if miss['ret_m']==0 and miss['mktcap_6']==0 and miss['bm']==0 else 'FAIL'}] no missing required fields")
w(f"  [{'OK' if (panel['mktcap_6']>0).all() else 'FAIL'}] no non-positive market cap")
w(f"  [{'OK' if panel['ret_m'].abs().max()<=T.MAX_MONTHLY_RETURN else 'FAIL'}] no implausible monthly returns in panel")
w(f"  [{'OK' if logs['integrity']['passed'] else 'FAIL'}] two-ME consistency + integrity")
w(f"  [OK] used_fund_date/walked present per row: "
  f"{panel['used_fund_date'].notna().all() and panel['walked'].notna().all()}")

OUT.close()
print("wrote scratch_pilot.txt")
