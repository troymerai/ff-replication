"""Diagnose extreme stitched returns + the split_bonus over-count in the 2010-06 pilot."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import ff_kr_transform as T
import ff_kr_extract as E

refs = T.reference_maps()
built = T.build_sort_cross_section("20100630", refs=refs)
adf = T.assign_portfolios(built["frame"], "kospi_only")["frame"]
ret = T.build_returns(adf.index.tolist(), "20100630", "20110630", delist=refs[1])

L = []
# 1) extreme returns: show cap/shares at both months
ext = ret[ret["ret_m"].abs() > 3.0].copy()
L.append(f"extreme rows (|ret|>3): {len(ext)}")
dates = T.month_end_dates("20100630", "20110630")
panel = T._cap_shares_panel(dates, T.MARKETS)
ordered = [d for d in dates if d in panel]
prevmap = {cur: prev for prev, cur in zip(ordered[:-1], ordered[1:])}
for r in ext.itertuples():
    cur = r.date.strftime("%Y%m%d")
    # find which resolved date maps to this month
    match = [d for d in ordered if pd.Timestamp(d).strftime("%Y-%m") == r.date.strftime("%Y-%m")]
    cur_d = match[0] if match else cur
    prev_d = prevmap.get(cur_d)
    try:
        c0 = panel[prev_d].at[r.ticker, "cap"]; s0 = panel[prev_d].at[r.ticker, "shares"]
        c1 = panel[cur_d].at[r.ticker, "cap"]; s1 = panel[cur_d].at[r.ticker, "shares"]
        L.append(f"  {r.ticker} {prev_d}->{cur_d} ret={r.ret_m:+.2f} flag={r.adj_flag} "
                 f"cap {int(c0):,}->{int(c1):,} shares {int(s0):,}->{int(s1):,} "
                 f"k={s1/s0:.3f} cap_ratio={c1/c0:.3f}")
    except Exception as e:
        L.append(f"  {r.ticker} {cur}: lookup err {e}")

# 2) split_bonus k-distribution
sb = ret[ret["adj_flag"] == "split_bonus"]
# recover k per event
def kval(row):
    cur = row["date"].strftime("%Y%m%d")
    match = [d for d in ordered if pd.Timestamp(d).strftime("%Y-%m") == row["date"].strftime("%Y-%m")]
    cur_d = match[0] if match else cur
    prev_d = prevmap.get(cur_d)
    try:
        return panel[cur_d].at[row["ticker"], "shares"] / panel[prev_d].at[row["ticker"], "shares"]
    except Exception:
        return float("nan")
ks = sb.apply(kval, axis=1)
L.append("")
L.append(f"split_bonus events: {len(sb)}")
L.append(f"  k near 1 (|k-1|<0.05): {int((ks.sub(1).abs() < 0.05).sum())}")
L.append(f"  k in [0.05,0.5] band (real splits/bonus): {int(((ks.sub(1).abs()>=0.05) & (ks.sub(1).abs()<=0.5)).sum())}")
L.append(f"  k>1.5 or k<0.5 (large): {int(((ks>1.5)|(ks<0.5)).sum())}")
L.append(f"  k quantiles: {ks.quantile([0.01,0.25,0.5,0.75,0.99]).round(4).to_dict()}")

(ROOT / "scratch_retdiag.txt").write_text("\n".join(L), encoding="utf-8")
print("done")
