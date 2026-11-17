"""Frozen pattern tests on captured Hyperliquid hourly bars.

The ten tests below were written down on 2026-11-17 before their counts were read, and the
baseline counts are in LESSONS.md (method note of that date). The point of this file is that the
tests do not change: run it on bars captured later and compare. A new idea is added at the bottom
of CANDIDATES with its date and is scored only on bars after that date.

Usage:
    python hl_patterns.py --archive <data_dir>/archive/candles [--since 2026-11-17T10:00Z] [--crypto]

--since keeps only hours at or after that time (the out-of-sample view).
--crypto also pulls 5,000 hourly bars for BTC, ETH, HYPE and SOL from the public API and runs the
weekend and follow-through tests on them. Nothing is written to disk. Never commit the data.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FROZEN = "2026-11-17T10:00Z"
US = (
    "AAPL AMD AMZN AVGO BABA COIN COST CRCL CRWV GOOGL HOOD INTC LLY META MSFT MSTR MU NFLX NVDA "
    "ORCL PLTR TSLA TSM RIVN GME IBM QCOM MRVL DELL SNDK ARM ASML LRCX AMAT SMH SOXL MAGS XLE TLT"
).split()
IDX = ["SP500", "XYZ100"]
CANDIDATES: list[tuple[str, str]] = [
    # (date added, one-line definition a script can apply). Scored only on later bars.
]
API = "https://api.hyperliquid.xyz/info"


def tstat(x) -> float | None:
    x = pd.Series(x).dropna()
    return round(x.mean() / x.std() * np.sqrt(len(x)), 2) if len(x) > 2 and x.std() > 0 else None


def load_archive(root: str, min_rows: int):
    px, vol, fund, prem = {}, {}, {}, {}
    for n in US + IDX:
        try:
            d = pd.read_parquet(f"{root}/xyz_{n}/1h.parquet", columns=["t", "close", "volume"])
            d = d.drop_duplicates("t").set_index("t").sort_index()
            if len(d) < min_rows:
                continue
            f = pd.read_parquet(f"{root}/xyz_{n}/funding.parquet").drop_duplicates("t")
            f = f.set_index("t").sort_index()
        except Exception:
            continue
        px[n], vol[n] = d["close"], d["close"] * d["volume"]
        fund[n], prem[n] = f["funding_rate"], f["premium"]
    p = pd.DataFrame(px)
    full = pd.date_range(p.index[0], p.index[-1], freq="1h")
    return (
        p.reindex(full).ffill(),
        pd.DataFrame(vol).reindex(full).fillna(0),
        pd.DataFrame(fund).reindex(full),
        pd.DataFrame(prem).reindex(full),
    )


def post(body: dict):
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def load_crypto(coins=("BTC", "ETH", "HYPE", "SOL")):
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start = end - 5000 * 3600 * 1000
    px, fund = {}, {}
    for c in coins:
        k = post({"type": "candleSnapshot", "req": {"coin": c, "interval": "1h", "startTime": start, "endTime": end}})
        s = pd.Series({pd.Timestamp(x["t"], unit="ms", tz="UTC"): float(x["c"]) for x in k})
        px[c] = s[~s.index.duplicated()].sort_index()
        rows, t0 = [], start
        while t0 < end:
            h = post({"type": "fundingHistory", "coin": c, "startTime": t0, "endTime": end})
            if not h:
                break
            rows += h
            t0 = h[-1]["time"] + 1
            if len(h) < 500:
                break
        f = pd.Series({pd.Timestamp(x["time"], unit="ms", tz="UTC").floor("h"): float(x["fundingRate"]) for x in rows})
        fund[c] = f[~f.index.duplicated()].sort_index()
    p = pd.DataFrame(px)
    full = pd.date_range(p.index[0], p.index[-1], freq="1h")
    return p.reindex(full).ffill(), pd.DataFrame(fund).reindex(full)


def masks(index):
    et = index.tz_convert("America/New_York")
    h, wd = et.hour, et.weekday
    wk = ((wd == 4) & (h >= 20)) | (wd == 5) | ((wd == 6) & (h < 18))
    cash = (h >= 10) & (h < 16) & (wd < 5)
    ses = np.where(
        wk,
        "weekend",
        np.where(
            cash,
            "cash 10-16",
            np.where(
                (h == 9) & (wd < 5),
                "open hour 9-10",
                np.where((h >= 4) & (h < 9), "pre 4-9", np.where((h >= 16) & (h < 20), "post 16-20", "overnight 20-4")),
            ),
        ),
    )
    return et, pd.Series(wk, index=index), pd.Series(cash, index=index), pd.Series(ses, index=index)


def weekend_table(logp: pd.DataFrame, cols: list[str], since) -> pd.DataFrame:
    et = logp.index.tz_convert("America/New_York")
    rows = []
    for d in pd.date_range(et[0].normalize(), et[-1].normalize(), freq="W-FRI"):
        t = [d + pd.Timedelta(hours=19), d + pd.Timedelta(days=2, hours=17), d + pd.Timedelta(days=3, hours=8)]
        t = [x.tz_convert("UTC") for x in t]
        if since is not None and t[0] < since:
            continue
        if not all(x in logp.index for x in t):
            continue
        a, b, c = (logp.loc[x, cols] for x in t)
        rows.append({"fri": d.date(), "weekend": 100 * (b - a).mean(), "sunday_open_to_mon9": 100 * (c - b).mean()})
    return pd.DataFrame(rows)


def follow_through(r: pd.DataFrame, cols, mask, label) -> None:
    ac, big = [], []
    for k in cols:
        x, nx = r[k], r[k].shift(-1)
        m = mask & x.notna() & nx.notna()
        if m.sum() < 50:
            continue
        ac.append(x[m].corr(nx[m]))
        bm = m & (x.abs() >= 2 * x[m].std())
        big += list((np.sign(x[bm]) * nx[bm] * 100).values)
    if not big:
        print(f"  T3 {label}: too few hours")
        return
    big = np.array(big)
    print(
        f"  T3 {label}: lag-1 autocorrelation {np.mean(ac):+.3f}; after a 2-sigma hour (n={len(big)}) "
        f"next hour reverses {100 * (big < 0).mean():.0f} %, mean signed {big.mean():+.3f} %"
    )


def report_weekend(w: pd.DataFrame, label: str) -> None:
    if w.empty:
        print(f"  T2 {label}: no complete weekend in range")
        return
    a, b = w["weekend"], w["sunday_open_to_mon9"]
    same = int((np.sign(a) == np.sign(b)).sum())
    print(
        f"  T2 {label}: Fri 20:00 ET -> Sun 18:00 ET up {int((a > 0).sum())} of {len(w)}, mean {a.mean():+.2f} %, "
        f"t {tstat(a)} | Sun 18:00 -> Mon 09:00 up {int((b > 0).sum())} of {len(w)}, mean {b.mean():+.2f} % | "
        f"same sign {same} of {len(w)}"
    )


def run_equities(root: str, since, min_rows: int) -> None:
    p, v, f, prem = load_archive(root, min_rows)
    logp = np.log(p)
    r = logp.diff()
    stocks = [c for c in p.columns if c not in IDX]
    et, wk, cash, ses = masks(p.index)
    keep = pd.Series(True, index=p.index) if since is None else pd.Series(p.index >= since, index=p.index)
    print(f"stock and index contracts: {len(p.columns)} markets, hours in view {int(keep.sum())}, "
          f"{p.index[keep][0]} to {p.index[-1]}")
    if "SP500" in p:
        g = pd.DataFrame({"r": r["SP500"], "s": ses, "v": v["SP500"]})[keep].dropna()
        out = g.groupby("s").agg(hours=("r", "size"), sum_ret_pct=("r", lambda x: round(100 * x.sum(), 1)),
                                 var=("r", lambda x: (x**2).sum()), vol=("v", "sum"))
        out["var_share_pct"] = (100 * out["var"] / out["var"].sum()).round(1)
        out["vol_share_pct"] = (100 * out["vol"] / out["vol"].sum()).round(1)
        big = g[g.r.abs() >= 0.005].groupby("s").size().reindex(out.index).fillna(0)
        out["big_hours"] = big.astype(int)
        out["big_rate_pct"] = (100 * big / out["hours"]).round(1)
        print("  T1/T7 S&P contract by session")
        print(out.drop(columns=["var", "vol"]).to_string())
    report_weekend(weekend_table(logp, stocks, since), "stocks, equal weight")
    for k in IDX + ["NVDA"]:
        if k in p:
            report_weekend(weekend_table(logp, [k], since), k)
    night = []
    for d in pd.date_range(et[0].normalize(), et[-1].normalize(), freq="B"):
        t0 = (d + pd.Timedelta(hours=19)).tz_convert("UTC")
        t1 = (d + pd.Timedelta(days=1, hours=3)).tz_convert("UTC")
        if d.weekday() != 4 and t0 in logp.index and t1 in logp.index and (since is None or t0 >= since):
            night.append(100 * (logp.loc[t1, stocks] - logp.loc[t0, stocks]).mean())
    night = pd.Series(night)
    if len(night):
        print(f"  T10 control, weekday nights 20:00 -> 04:00 ET: up {int((night > 0).sum())} of {len(night)}, "
              f"mean {night.mean():+.3f} %")
    for label, m in (("cash hours", cash & keep), ("weekday off-hours", ~cash & ~wk & keep), ("weekend", wk & keep)):
        follow_through(r, stocks, m, f"stocks, {label}")
    same = tot = bsame = btot = 0
    for d in pd.date_range(et[0].normalize() + pd.Timedelta(days=1), et[-1].normalize(), freq="B"):
        if d.weekday() == 0:
            continue
        t = [d - pd.Timedelta(days=1) + pd.Timedelta(hours=15), d + pd.Timedelta(hours=8), d + pd.Timedelta(hours=15)]
        t = [x.tz_convert("UTC") for x in t]
        if not all(x in logp.index for x in t) or (since is not None and t[0] < since):
            continue
        on = (logp.loc[t[1], stocks] - logp.loc[t[0], stocks]) * 100
        day = (logp.loc[t[2], stocks] - logp.loc[t[1], stocks]) * 100
        ok = on.notna() & day.notna() & (on != 0)
        same += int((np.sign(on[ok]) == np.sign(day[ok])).sum())
        tot += int(ok.sum())
        b = ok & (on.abs() >= 2)
        bsame += int((np.sign(on[b]) == np.sign(day[b])).sum())
        btot += int(b.sum())
    if tot:
        print(f"  T4 day session keeps the overnight sign: {same} of {tot} ({100 * same / tot:.1f} %); "
              f"after a 2 % overnight move {bsame} of {btot}")
    fs = f[stocks][keep]
    vals = fs.values[~np.isnan(fs.values)]
    if len(vals):
        print(f"  T5 funding, stocks: mean {vals.mean() * 24 * 365 * 100:+.1f} % a year, positive in "
              f"{100 * (vals > 0).mean():.0f} % of hours")
    c16 = logp[[(x.hour == 15) and (x.weekday() < 5) for x in et] & keep.values][stocks]
    dr = c16.diff().dropna(how="all") * 100
    sp = []
    for i in range(len(dr) - 1):
        row = dr.iloc[i].dropna().sort_values()
        nx = dr.iloc[i + 1]
        if len(row) >= 10:
            sp.append(nx[row.index[-5:]].mean() - nx[row.index[:5]].mean())
    if len(sp) > 2:
        sp = np.array(sp)
        print(f"  T6 yesterday's five best less five worst, next day: mean {sp.mean():+.3f} %, positive "
              f"{int((sp > 0).sum())} of {len(sp)}")
    for hz in (24, 72):
        lower_p = lower_f = n = 0
        for k in p.columns:
            fw = ((logp[k].shift(-hz) - logp[k]) * 100)[keep]
            z, fk = prem[k][keep], f[k][keep]
            m = z.notna() & fw.notna() & fk.notna()
            if m.sum() < 200:
                continue
            n += 1
            lower_p += fw[m][z[m] >= z[m].quantile(0.9)].mean() < fw[m][z[m] <= z[m].quantile(0.1)].mean()
            lower_f += fw[m][fk[m] >= fk[m].quantile(0.95)].mean() < fw[m].mean()
        if n:
            print(f"  T8/T9 {hz}h: return lower after a rich premium in {lower_p} of {n} markets; lower after "
                  f"top-5 % funding in {lower_f} of {n}")


def run_crypto(since) -> None:
    p, f = load_crypto()
    logp = np.log(p)
    r = logp.diff()
    _, wk, _, _ = masks(p.index)
    keep = pd.Series(True, index=p.index) if since is None else pd.Series(p.index >= since, index=p.index)
    print(f"crypto contracts: {list(p.columns)}, {p.index[0]} to {p.index[-1]}")
    for k in p.columns:
        report_weekend(weekend_table(logp, [k], since), k)
        fk = f[k][keep].dropna()
        if len(fk):
            print(f"  T5 {k} funding: mean {fk.mean() * 24 * 365 * 100:+.1f} % a year, positive in "
                  f"{100 * (fk > 0).mean():.0f} % of hours")
    follow_through(r, list(p.columns), wk & keep, "crypto, weekend")
    follow_through(r, list(p.columns), ~wk & keep, "crypto, weekdays")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", help="path to archive/candles")
    ap.add_argument("--since", help="ISO time; keep only hours at or after it")
    ap.add_argument("--crypto", action="store_true")
    ap.add_argument("--min-rows", type=int, default=3000)
    a = ap.parse_args()
    since = pd.Timestamp(a.since).tz_convert("UTC") if a.since else None
    print(f"tests frozen {FROZEN}; view: {'all bars' if since is None else 'bars since ' + str(since)}")
    if a.archive:
        run_equities(a.archive, since, a.min_rows)
    if a.crypto:
        run_crypto(since)
    for date, text in CANDIDATES:
        print(f"candidate {date}: {text}")


if __name__ == "__main__":
    main()
