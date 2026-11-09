# TAPE call — SK hynix (KRX 000660), Mon 2026-11-09 KRX close vs Fri 2026-11-06 close — opinion at the user's request, recorded so it can be graded

Made 2026-11-08 ~15:00 UTC (Sunday in Seoul). TAPE class, third entry. Switch probabilities set from counts where a count exists, per the Nov 8 method note.

## Measured (M)
- Last six sessions (KRX closes): Oct 30 −7.1 %, Nov 2 +3.8 %, Nov 3 −5.0 %, Nov 4 −0.1 % (intraday −6.6 % at the low, bought back), **Nov 5 −9.5 %, Nov 6 −10.4 %** (opened −13.8 % at ₩1,089,000, closed ₩1,132,000 — four points bought back from the open). **61 % below the 252-day close high (₩2,919,000); −34 % over 21 sessions.** 21-day daily sigma 5.9 %.
- Friday Nov 6 in the US, after Seoul closed: SOXX +3.51 %, **MU +5.1 %, SNDK +3.6 %**, WDC −1.4 %, NVDA +2.8 %; **EWY (the US-listed Korea ETF) +4.0 %** after the Kospi's −7 % Friday — the closest measured proxy for where Seoul opens Monday.
- Own conditionals from the 5-year daily history (counts, not priors): after a day of −7 % or worse, the next session closed up **22 of 36** (median +1.1 %); after two consecutive down days (≤ −3 % then ≤ −5 %), up **10 of 15** (median +3.0 %); after a three-day drop of 15 % or more, up **8 of 12** (median +3.1 %). Friday was all three.
- Ledger: the one hynix print graded this season (Oct 28) was bought +10.9 % from a 55 % drawdown on a bouncing tape after a 21 % index crash — the "lopsided own cell after a drawdown" case.

## Derived (D)
- The Friday close above the open (+4 points from the low) says the forced selling exhausted inside the session, the same shape as SOXL's Friday in the US. EWY +4 % is Friday's US session pricing Monday's Seoul open; it has been right on sign both times this week (Thursday's US crash → Friday Seoul −7 %; it now says up).
- The memory complex specifically bounced in the US (MU, SNDK) while hard-drive WDC did not: the bid is in the names hynix trades with.

## Kill switches, with counted probabilities
| # | Trigger | P (source) | Consequence |
|---|---|---|---|
| KS1 | Monday open above ₩1,132,000 | 0.70 (EWY +4 % Friday; own next-day-up rate after a ≤ −7 % day 0.61) | → the up case; the open's sign has held to the close on the two Seoul sessions this week |
| KS2 | Monday open ≤ −3 % | 0.20 (two of the last six sessions opened that low) | side lost unless bought back as on Nov 4 and Nov 6 |
| KS3 | US index futures ≤ −1.5 % at the Seoul open (a weekend catalyst for the capex-scare story) | 0.15 (no dated event; H) | P(up) −0.15 |
| KS4 | Samsung (005930) up with it | 0.70 (same tape) | confirms |
| KS5 | Logistics (settled row, KRX holiday — none scheduled) | 0.03 | re-grade |

## The call (H on the weight, M on the counts)
**SK hynix closes Monday up about 4 %, near ₩1,177,000.** P(up) 0.67 — the own conditional (0.61-0.67) lifted by EWY's Friday sign, cut by the weekend-catalyst term. Size: the two-down-day and three-day-drop medians (+3.0, +3.1) raised toward the 5.9 % daily sigma because Friday's buy-back from the open shows the bid is present.

## What would disprove it
A Monday open below Friday's close of ₩1,132,000, or Samsung opening red with it: the Friday US bounce not transmitting, which would mean the Korean selling is its own story (memory pricing, the Kospi's leverage) rather than the AI-tape echo I am reading it as.

## Grading (2026-11-09 07:15 UTC; regular-session close PROVISIONAL — vendors disagree while the KRX after-hours session trades)
- **Closes (M):** Fri reference ₩1,132,000. **Mon Nov 9 regular-session close ₩1,136,000 (stockanalysis row: open 1,130,000, high 1,193,000, low 1,116,000) → r = +0.35 %.** Naver and Daum at 16:10 KST show ₩1,124,000 (−0.71 %) with marketStatus OPEN — the KRX after-hours single-price session, not the 15:30 close; Samsung shows the same pattern (stockanalysis regular close 189,900 +0.64 %; Naver 187,200 −0.79 % after hours). Grade on the regular close; confirm at the 21:40 UTC wake from a dated daily row (Daum days API). If the dated row reads below ₩1,132,000 the sign flips to a miss.
- **Session shape (M):** open −0.18 % (KS1 "open above Friday's close" **not fired**, by 0.18 %; KS2 not fired), low −1.4 % in the first hour, **high +5.4 % (₩1,193,000) mid-session** — the Nov 4 / Nov 6 buy-back shape, reaching the called +4 % — then a fade into the close to +0.35 % and about −0.7 % in the after-hours session. Samsung +0.64 % regular close (**KS4 confirmed**). Japan's semis closed up (Advantest +1.3 %, Tokyo Electron +2.8 %).
- **The exogenous term (M, Hyperliquid perps vs Fri 20:00 UTC):** at the Seoul open (00:00 UTC) NVDA perp −3.9 %, SOXL perp −4.3 %, SKHY −2.1 %, KR200 −1.6 %, SP500 perp flat → **KS3 fired on the sector leader, not on the index.** At 07:00 UTC: NVDA −5.7 %, SOXL −5.1 %, SKHY −2.6 %. The Asian semis tape closed green with the leader's perp down 4-6 %: the trust shock did not transmit to Asia's chips during the session.
- **Direction: UP called, +0.35 % realised → HIT on the sheet (provisional), a no-information outcome in the lessons** (|r| inside the ±1 % band that the ledger treats as noise). TAPE class 2 of 2 on sign, size error 3.65 pts.
- **Counted switches, first graded use:** the next-day-up conditionals (0.61-0.67) were consistent with the outcome; the size medians (+3.0 / +3.1) were reached intraday (+5.4 at the high) and not held — on a day with a −4 % sector-leader shock the counted median applies to the high, not the close. EWY's Friday +4.0 % predicted the open's sign wrongly by 0.18 %, i.e. it predicted flat; it was right that Seoul would not follow the US Friday bounce fully. The buy-back-from-the-open signature appeared (low −1.4 → high +5.4) and then failed into the close, which is new: on the two prior instances the buy-back held.
- **Lesson (n = 1):** when an exogenous shock lands on the sector leader between the lock and the open, the counted conditional still gives the sign, but the size should be cut to the shrink line (1 %) — the shock takes the afternoon. Disproof: the next such case closing at or beyond the counted median.

- **Re-stamp, 2026-11-09 21:42 UTC (M):** the dated daily rows now agree. Daum days API: Nov 9 open 1,130,000, high 1,193,000, low 1,116,000, **close 1,136,000 (+0.353 %)**; stockanalysis the same. The provisional flag is lifted; **HIT on sign stands**, a no-information outcome as written.

## Pre-open update
- **Pre-open note, Sun 2026-11-08 ~15:00 UTC (M/S; observation, the call stands as locked):** a weekend catalyst arrived. Press reports dated Nov 8 (Axios, CNN affiliates) say Nvidia's CEO admitted, in a message to staff, a personal relationship with a person US officials suspect of working for Chinese intelligence; the board has opened an independent-committee investigation and says he continues as CEO; on Nov 6 three House committee chairmen had already asked him to address his interactions with an executive of a company named in the reports, with a reply due Nov 11 and a briefing proposed for Nov 18. The intelligence claim rests on unnamed officials and is not confirmed by the company (S). **Nvidia shares have not yet traded on it; Monday is the first session.** This is the undated exogenous term the document did not carry: it enters the Monday session as a sector-leader shock, sign negative for the AI tape at the open, size unknown. For this call it is KS3's case (a weekend catalyst with US futures likely lower at the Seoul open; stated P 0.15, now realised) → P(up) moves from 0.67 toward 0.52 on the document's own rule; the graded line stays UP ≈ +4 %.

- **Sun 2026-11-08 22:00 UTC (M, Hyperliquid 24/7 perpetual markets, 1-hour candles vs the Friday 20:00 UTC bar):** the first prices on the weekend news. **NVDA perp $182.22, −2.4 %**; **SOXL perp $44.01, −4.1 %**; **SK hynix perp (xyz:SKHY) $125.68, −1.2 %** (−2.5 % from its Saturday-Sunday plateau near $128.9); Samsung perp −1.0 %; KR200 perp −1.0 %; EWY perp −0.5 %. All six were flat through Saturday and Sunday daytime and dropped in the 16:00-22:00 UTC Sunday window, i.e. on the news, not on carry. The exogenous term's first measured size is therefore a 2-4 % hit to the AI-leader and leveraged-semis proxies and a 1 % hit to the Korea proxies — a shock, not a crash, as priced seven hours before Seoul opens. For this call: KS3 (US futures ≤ −1.5 % at the Seoul open) is on course to fire on the NVDA and SOXL proxies; the hynix proxy's own −1.2 % says the Monday gap is small-negative, so the called +4 % now requires a reversal inside the session (the Nov 4 and Nov 6 buy-back shape). P(up) on the document's rule: 0.52. The graded line stays UP ≈ +4 %.

CALL: SK hynix UP ≈ +4 % (P(up) 0.67) — Nov 9 KRX close vs Nov 6 close ₩1,132,000.
