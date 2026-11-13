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
move, some tick, and a few jump. The uncertainty of a market's future
price is captured here by the recent size of its own one-minute moves.

1. Volatility from two windows. For each market we take the mean
   absolute deviation from the mean (MAD) of its last W one-minute
   changes, for W = 120 and W = 720 (using every change when the series
   is shorter). A single 120-minute window is a noisy read of a
   possibly quiet stretch: when a quiet stretch misrepresents the recent
   regime it reads the market as nearly certain, which is badly wrong at
   the far horizons. Pooling a long 720-minute window keeps the far
   horizons honest by anchoring them to a slower, steadier estimate; the
   short 120-minute window still lets a genuinely more active market
   widen. The two are mixed as sigma = min(0.004, max(0.0002,
   0.7 * mad120 + 0.3 * mad720)): the 0.7/0.3 weighting leans on the
   recent window but never throws away the long one, the 0.0002 floor
   avoids a degenerate zero sigma from a literally motionless series,
   and the 0.004 ceiling keeps an outlier-heavy window from exploding
   the bands.

2. sqrt(h) scaling. Over h minutes the price moves as a sum of h
   roughly independent one-minute steps, so its spread grows like
   sqrt(h); this is the standard random-walk scaling and gives the far
   horizons their wider bands.

3. Forecast. For horizon h = 1..240 and quantile tau, the prediction is
   clip(x + z_tau * spread_tau * sigma * sqrt(h), 0, 1) with
   z = -1.644854, -0.674490, 0.0, 0.674490, 1.644854 for tau = 0.05,
   0.25, 0.5, 0.75, 0.95 (Gaussian quantiles), every value rounded to 6
   decimals. The per-quantile SPREADS widen the 0.05 and 0.95 bands by
   1.25 and the 0.25 and 0.75 bands by 1.1, to account for the fat tails
   of jump-driven price processes; the clip to [0, 1] keeps the
   forecasts inside the price domain.

A market with fewer than 5 prices, or any non-finite price, falls back
to sigma = 0.001 for that market instead of failing.

Everything is a deterministic function of the case file: no clock, no
randomness, no state between runs, so two runs write identical bytes.

Verification run on the kit's practice cases, one case per process
exactly as the grader runs them: every output passes strict format
validation (one row per market of the case, 240 rows of five finite,
non-decreasing quantiles in [0, 1]), and the run over MAKEFUN_INPUT_DIR
completed clean, writing valid output for every case file.
"""

import json
import math
import os
import sys
from pathlib import Path

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

W_SHORT = 120          # minutes in the short volatility window
W_LONG = 720           # minutes in the long volatility window
SIGMA_FLOOR = 0.0002   # never treat a market as perfectly certain
SIGMA_CAP = 0.004      # never let an outlier-heavy window explode the bands
SPREADS = [1.25, 1.1, 1.0, 1.1, 1.25]  # per-quantile band half-width: the tails of a price move are fatter than the Gaussian
FALLBACK_SIGMA = 0.001  # used when a market has too few / bad prices


def mad_of_changes(prices, window):
    """Mean absolute deviation from the mean of the last `window` 1-min changes."""
    changes = [b - a for a, b in zip(prices, prices[1:])]
    if not changes:
        return 0.0
    tail = changes[-window:] if len(changes) > window else changes
    mean = sum(tail) / len(tail)
    return sum(abs(c - mean) for c in tail) / len(tail)


def market_sigma(prices):
    """Two-window volatility estimate, or the fallback when unusable."""
    if len(prices) < 5:
        return FALLBACK_SIGMA
    for v in prices:
        if not math.isfinite(v):
            return FALLBACK_SIGMA
    mad120 = mad_of_changes(prices, W_SHORT)
    mad720 = mad_of_changes(prices, W_LONG)
    sigma = 0.7 * mad120 + 0.3 * mad720
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))


def forecast_case(case):
    """{market_id: [horizon rows of 5 quantiles]} for one parsed case."""
    horizon = int(case["horizon_minutes"])
    markets = case["markets"]

    out = {}
    for m in markets:
        prices = m["prices"]
        x = prices[-1] if prices else 0.5
        sigma = market_sigma(prices)
        rows = []
        for h in range(1, horizon + 1):
            row = []
            for i, z in enumerate(Z):
                scale = SPREADS[i] * sigma * math.sqrt(h)
                v = round(min(1.0, max(0.0, x + z * scale)), 6)
                row.append(0.0 if v == 0 else v)  # never -0.0
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
