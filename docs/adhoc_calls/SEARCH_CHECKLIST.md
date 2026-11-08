# Pre-lock search checklist — categories, not items

The human's role is to edit this list, never to supply what the searches find. A category is added when a graded miss shows a class of information that existed before the lock and that the searches did not reach. Each category names the miss that created it. The call document records, per category, what was searched and what was found or not found; "not read" is an allowed and required answer.

| # | Category | Searched how | Created by |
|---|---|---|---|
| 1 | The bar: company guide, prior-quarter exhibit, street consensus, release clock | EDGAR exhibits, TDnet, the company notice | the original method |
| 2 | Documents beyond the headline exhibit: presentation deck, Q&A, supplementary PDF, the 10-Q | the company's IR page and the second 6-K / 8-K of the day | Sony (quake clause unscorable, Nov 6) |
| 3 | Unnumbered lines in management language: an investment, a gap, a "we will update" | the prior two releases and the call transcript if readable, read for sentences without numbers | DraftKings (Predictions spend, Nov 6) |
| 4 | Sell-side changes to the bar in the last 30 days: cuts, downgrades, the reason given | one search per name on "cut" / "downgrade" with the month | DraftKings (Jefferies cut), Novartis / Merck (street-lowered bars) |
| 5 | People, regulators, governments: executive investigations, letters from committees, lawsuits, export rules — on the name and on the sector's largest company | three searches, last five days, no financial terms in the query | Nvidia CEO letter of Nov 6 (missed on the Nov 7-8 locks) |
| 6 | The index's own dated events between lock and grade: CPI, payrolls, FOMC, option expiry | the week-ahead calendar | the Nov 5-6 index days |
| 7 | The sibling that reported first: what line it was paid or punished for | the sibling's own exhibit and close | Expedia → Airbnb (misread), UPS → FedEx |
| 8 | What the social feeds carry that search engines do not | not reachable from this session; a human flag is the only source | (open) |
| 9 | Weekend and overnight prices on the 24/7 Hyperliquid perpetual markets for the name, the sector leader, SOXL and the country index (xyz:SKHY, xyz:SMSN, xyz:NVDA, xyz:SOXL, xyz:KR200, xyz:EWY, xyz:SP500 …) — the harness archives them and the public info endpoint serves 1-hour candles without a key | one candleSnapshot request per market, compared to the Friday 20:00 UTC bar | the Nov 8 Nvidia news: the markets priced it seven hours before Seoul opened and the Sunday locks did not read them |

Rules. The checklist is run at every lock and its results are written into the document under "Searched". A human may add a row by naming the miss and the class; a human does not add items under a row before a lock. Items the human does send before a lock are recorded as human leads with their source and graded separately in `CALIBRATION.csv`.
