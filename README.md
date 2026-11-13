# Pooled-shape quantile forecaster — make.fun polymarket-nvidia-ceo-forecast-01

A deterministic forecasting program (Python 3.13, standard library
only, no network) for Polymarket's "Next CEO of NVIDIA?" event. For
each case it reads the minute-by-minute Yes prices of every market in
the case (72 hours up to the checkpoint) and writes, for each market
and each of the 240 minutes after the checkpoint, five quantiles of
that minute's price (0.05, 0.25, 0.5, 0.75, 0.95), scored by mean
pinball loss.

## How it works

Prediction-market prices are sticky step processes: most minutes do
not move, some tick, a few jump. Instead of one Gaussian sigma for
every market (the house baseline), the forecaster pools the whole
case:

1. **Scale** — each market gets σ_m, the RMS of its 1-minute price
   changes over its full history in the case.
2. **Shape** — for each horizon bucket (1…240 min), every market's
   price changes over the last 1,080 minutes are standardized by
   σ_m·√h and pooled across markets; the pooled quantiles are the
   shared shape of an h-minute move (a large atom at zero, fat tails).
3. **Idiosyncrasy** — each market's own standardized quantiles are
   blended in (weight 0.3), and its tail quantiles are re-scaled by
   the ratio of its own 5%/95% quantiles to the pool's at h = 100,
   clipped to [0.8, 1.6]: headline candidates jump harder than the
   pool, quiet markets less.
4. **Forecast** — quantiles interpolated in √h between buckets,
   offsets narrowed ×0.9 around the median, scaled by σ_m·√h around
   the checkpoint price, clamped to [0, 1], rounded to 6 decimals.

Pure function of the case file: no clock, no randomness, no state —
two runs write identical bytes. Backtested on the kit's practice
cases (answers reconstructed from the following case's input):
**0.3576** mean pinball loss on p00–p04 vs **0.3925** for the house
baseline on the same subset. Slowest practice case runs in ~0.8 s of
the 8 s cap.

## Run

The program reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default
`/output`):

```
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Entrant: @urelkdubdqwr · License: MIT
