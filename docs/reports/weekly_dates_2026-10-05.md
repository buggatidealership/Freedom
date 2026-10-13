# Weekly earnings-date verification, 2026-10-05 (reduced: repository not restored)

The container was reset on Oct 2 and the GitHub account is still suspended (push and clone 403), so
the harness code, the local events table and the SEC bundle were unavailable. Steps done: the
vendor calendars (FMP and Finnhub, one request each, Oct 6-19; payloads kept in the scratchpad, not
committed), the Nasdaq calendar as a third source, and first-hand issuer checks. Not done: the
freedom-data artifact, `uv run freedom sec-bundle`, ruff/pytest, overrides, commit, push.

Universe: the 120 equity-like Hyperliquid markets listed on Oct 5 (xyz, para, io dexes).

| Event | Vendor dates | Issuer source (read Oct 5) | Harness action |
|---|---|---|---|
| ASML Q3 2026 | FMP 10-14, Nasdaq 10-14 pre-market, Finnhub none | Statutory interim report 6-K (accession 0001628280-26-048235): "Financial calendar October 14, 2026 Announcement of third quarter results for 2026"; clock not stated (habit 07:00 CET) | none: `release_clock_overrides` ASML 07:00 Europe/Amsterdam and `t0_overrides` ASML:2026-10-14 05:00Z already match |
| TSMC Q3 2026 (TSM ADR) | FMP 10-15, Nasdaq 10-15 pre-market, Finnhub none | investor.tsmc.com financial calendar: "October 15, 2026 TSMC 3Q'26 Results - Earnings Conference and Conference Call 14:00-15:30 Asia/Taipei" | none: `t0_overrides` TSM:2026-10-15 06:00Z already matches |
| Blackstone Q3 2026 (BX) | Finnhub 10-15 bmo; FMP 10-22; Nasdaq not listed in the window | Company notice "Blackstone Announces Third-Quarter 2026 Investor Call" (blackstone.com/?p=48748, blocked to our fetches; search summary: call Oct 22, 2026 at 9:00 am ET); Earnings Whispers "confirmed ... October 22, 2026, before the market open"; MarketScreener 22/10/2026. 8-K item 2.02 acceptances 2025-10-23 10:57Z, 2026-01-29 11:55Z, 2026-04-23 10:55Z, 2026-07-23 10:55Z (about 6:55 am ET) | Finnhub's 10-15 is wrong; FMP's 10-22 is right, so no override is needed if the harness takes FMP's date. Outside the 14-day window anyway. First-hand quote of the notice still to be read when blackstone.com is reachable. |

Vendor disagreements left open: Finnhub BX 10-15 vs FMP 10-22 (issuer: 10-22). Finnhub lists no ASML or
TSM in the window (symbol coverage), FMP lists no BX in the window (correctly).
No other Hyperliquid equity reports between Oct 6 and Oct 19 on any of the three calendars. The
next cluster is Oct 20-29 (NFLX 10-20, IBM and LRCX 10-21, INTC and BX 10-22, TER 10-27, GOOGL,
META, MSFT, NOW, STX, TSLA 10-28, AAPL, AMZN, COIN, WDC 10-29 per FMP, all unverified against the
issuers; next week's check covers them).
