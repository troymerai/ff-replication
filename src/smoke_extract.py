"""
E-layer smoke — one rebalance date. Confirms E1-E4 return joinable cross-sections
via the resolver (not the full panel; just proves the pieces fit and the walk-back
fires on the year-end date).

Run: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\src\smoke_extract.py"
"""
import ff_kr_extract as E          # sibling module (same folder)

SORT_DATE = "20100630"             # end-June sort: size uses this June ME
DEC_DATE  = "20091231"             # prior Dec: B/M uses this ME + book (Dec 31 is non-trading -> walk-back)

for market in E.MARKETS:
    print(f"\n===== {market} =====")
    univ = E.e1_universe(SORT_DATE, market)
    print(f"E1 universe @ {SORT_DATE}: {len(univ)} tickers")

    fund = E.e2_fundamentals(DEC_DATE, market)     # B/M from prior Dec
    cap6 = E.e3_marketcap(SORT_DATE, market)       # size from June
    px6  = E.e4_prices(SORT_DATE, market)          # close from June

    skip = f" skip={fund.get('skip')}" if fund.get("skip") else ""
    print(f"E2 fundamentals @ {DEC_DATE}: ok={fund['ok']} used={fund.get('used')} walked={fund.get('walked')}{skip}")
    print(f"E3 market cap  @ {SORT_DATE}: ok={cap6['ok']} used={cap6['used']} walked={cap6['walked']}")
    print(f"E4 prices      @ {SORT_DATE}: ok={px6['ok']} used={px6['used']} walked={px6['walked']}")

    if not (fund["ok"] and cap6["ok"] and px6["ok"]):
        print("  -> a loader returned no data (expected for KOSDAQ if pre-onset); see above")
        continue

    # join on ticker index: B/M (prior Dec), size (June cap), close (June)
    f = fund["frame"][["BPS", "PBR"]]
    c = cap6["frame"][["시가총액", "상장주식수"]]
    px = px6["frame"]
    p = px[["종가"]] if "종가" in px.columns else px.iloc[:, [3]]

    j = f.join(c, how="inner").join(p, how="inner")
    j = j[(j["PBR"] > 0) & (j["시가총액"] > 0)]
    j["BM"] = 1.0 / j["PBR"]
    print(f"  joined (PBR>0 & cap>0): {len(j)} tickers"
          f" | B/M median={j['BM'].median():.3f}"
          f" | cap median={j['시가총액'].median():,.0f}")
    print(j.head(3).to_string())
