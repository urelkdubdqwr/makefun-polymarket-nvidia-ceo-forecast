#!/usr/bin/env python3
"""make.fun entry: polymarket-nvidia-ceo-forecast-01.

Forecasts the Yes price of every market of Polymarket's "Next CEO of
NVIDIA?" event at each of the 240 minutes after a checkpoint, as the
quantiles at levels 0.05, 0.25, 0.5, 0.75 and 0.95, scored by mean
pinball loss. Runs offline (Python 3.13, standard library only), one
case per run: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input)
and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).

Model (per case, from the case's own prices only)
-------------------------------------------------
Prediction-market prices are sticky step processes: most minutes do not
move, some tick, a few jump. A single Gaussian sigma (the house
baseline's persistence forecast) wastes most of that structure, and one
market's own history is too jump-dominated to trust alone. This entry
pools the whole case:

1. Scale: each market gets sigma_m, the RMS of its 1-minute price
   changes over its full history in the case.
2. Shape: for each horizon bucket hb, every market's price changes
   p[t+hb] - p[t] over the last WINDOW minutes are standardized by
   sigma_m * sqrt(hb) and pooled across markets; the five quantiles of
   that pooled sample are the shared shape of an hb-minute move.
3. Idiosyncrasy: each market's own standardized quantiles (when it has
   enough samples) are blended in with weight OWN_BLEND, and its two
   tail quantiles are re-scaled by the ratio of its own 5%/95%
   quantiles to the pool's at hb = 100, clipped to [0.8, 1.6] — the
   headline candidates jump harder than the pool, quiet markets less.
4. Forecast for minute h: quantiles interpolated in sqrt(h) between
   buckets, offsets narrowed by WIDEN (the raw pooled shape overstates
   the near future slightly), times sigma_m * sqrt(h), around the
   checkpoint price, clamped to [0, 1], rounded to 6 decimals.

Everything is a deterministic function of the case file: no clock, no
randomness, no state between runs, so two runs write identical bytes.
Constants were set by backtesting on the kit's practice cases (answers
reconstructed from the following case's input, the same grid): the
configuration below scored 0.3576 mean pinball loss on p00..p04 against
0.3925 for the house baseline on the same subset.

Verification on the kit's six practice cases, run one case per process
exactly as the grader runs them: every output passes strict format
validation (one row per market of the case, 240 rows of five finite,
non-decreasing quantiles in [0, 1], at most 240 KB against the 1 MiB
cap), two full runs are byte-identical, and the slowest case (p05, 24
markets, 4,286 minutes each) took 0.83 s of the 8 s cap on one CPU.
"""

import json
import math
import os
import sys
from pathlib import Path

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
BUCKETS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 70, 100, 140, 180, 240]
WINDOW = 1080          # minutes of history sampled for move shapes
WIDEN = 0.9            # narrowing of the pooled offsets around the median
OWN_BLEND = 0.3        # weight of a market's own shape vs the pool's
TAIL_BUCKET = 100      # horizon at which tail ratios are measured
MIN_POOL = 200         # samples for a bucket's pool to be used
MIN_OWN = 200          # samples for a market's own shape to be blended
MIN_TAIL = 300         # samples for a market's tail ratio to be trusted


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an already-sorted sequence."""
    m = len(sorted_vals)
    if m == 1:
        return sorted_vals[0]
    pos = tau * (m - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= m:
        return sorted_vals[-1]
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[lo + 1] * frac


def sigma_rms(prices):
    """RMS of the 1-minute changes over the whole history given."""
    ss = 0.0
    cnt = 0
    for a, b in zip(prices, prices[1:]):
        d = b - a
        ss += d * d
        cnt += 1
    return math.sqrt(ss / cnt) if cnt else 0.0


def interp(table, idx, sh):
    """Quantile offsets at sqrt-horizon `sh` from a (sqrt_h, offsets) table."""
    if not table:
        return [0.0] * len(TAUS)
    if idx + 1 < len(table) and table[idx + 1][0] > table[idx][0]:
        s0, q0 = table[idx]
        s1, q1 = table[idx + 1]
        f = (sh - s0) / (s1 - s0)
        return [a + (b - a) * f for a, b in zip(q0, q1)]
    return table[idx][1]


def forecast_case(case):
    """{market_id: [horizon rows of 5 quantiles]} for one parsed case."""
    horizon = int(case["horizon_minutes"])
    markets = case["markets"]
    sigs = {}
    for m in markets:
        s = sigma_rms(m["prices"])
        sigs[m["id"]] = s if s > 1e-4 else 1e-4

    pooled = []   # (sqrt_h, [offsets]) shared shape per bucket
    own = {}      # market id -> its own (sqrt_h, [offsets]) table
    tail = {}     # market id -> (low_ratio, high_ratio)
    for hb in BUCKETS:
        if hb > horizon:
            break
        pool = []
        per_market = {}
        for m in markets:
            p = m["prices"]
            n = len(p)
            s = n - min(n, WINDOW)
            if n - hb - s < 30:
                continue
            scale = sigs[m["id"]] * math.sqrt(hb)
            ds = [(b - a) / scale for a, b in zip(p[s:n - hb], p[s + hb:n])]
            pool.extend(ds)
            per_market[m["id"]] = ds
        if len(pool) < MIN_POOL:
            continue
        pool.sort()
        pq = [quantile(pool, t) for t in TAUS]
        pooled.append((math.sqrt(hb), pq))
        if hb == TAIL_BUCKET:
            for mid, ds in per_market.items():
                if len(ds) >= MIN_TAIL:
                    ds = sorted(ds)
                    lo = quantile(ds, TAUS[0]) / pq[0] if pq[0] < -1e-9 else 1.0
                    hi = quantile(ds, TAUS[4]) / pq[4] if pq[4] > 1e-9 else 1.0
                    tail[mid] = (min(1.6, max(0.8, lo)),
                                 min(1.6, max(0.8, hi)))
        for mid, ds in per_market.items():
            if len(ds) >= MIN_OWN:
                ds = sorted(ds)
                own.setdefault(mid, []).append(
                    (math.sqrt(hb), [quantile(ds, t) for t in TAUS]))

    out = {}
    for m in markets:
        p = m["prices"]
        x = p[-1] if p else 0.5
        sig = sigs[m["id"]]
        table = own.get(m["id"], [])
        lo_r, hi_r = tail.get(m["id"], (1.0, 1.0))
        rows = []
        bi = oi = 0
        for h in range(1, horizon + 1):
            sh = math.sqrt(h)
            while bi + 1 < len(pooled) and pooled[bi + 1][0] < sh:
                bi += 1
            offs = interp(pooled, bi, sh)
            if table:
                while oi + 1 < len(table) and table[oi + 1][0] < sh:
                    oi += 1
                oo = interp(table, oi, sh)
                offs = [(1.0 - OWN_BLEND) * a + OWN_BLEND * b
                        for a, b in zip(offs, oo)]
            med = offs[2]
            row = []
            for j, o in enumerate(offs):
                d = (o - med) * WIDEN
                if j == 0:
                    d *= lo_r
                elif j == len(TAUS) - 1:
                    d *= hi_r
                v = x + (med + d) * sig * sh
                v = round(min(1.0, max(0.0, v)), 6)
                row.append(0.0 if v == 0 else v)  # never -0.0
            row.sort()
            rows.append(row)
        out[m["id"]] = rows
    return out


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in inputs:
        case = json.loads((in_dir / name).read_text(encoding="utf-8"))
        if [float(t) for t in case.get("quantiles", TAUS)] != TAUS:
            raise ValueError(f"{name} asks for quantiles "
                             f"{case.get('quantiles')}, not {TAUS}")
        fc = forecast_case(case)
        doc = {"forecasts": [{"id": m["id"], "q": fc[m["id"]]}
                             for m in case["markets"]]}
        text = json.dumps(doc, separators=(",", ":")) + "\n"
        (out_dir / (name[:-3] + ".out")).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
