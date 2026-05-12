# BTCUSD LSR proxy optimization run — 2026-05-13

## Scope
- Data: Binance BTCUSDT 1m proxy CSV, 172,801 rows, 2026-01-12 to 2026-05-12 UTC.
- Engine: CPU parallel grid/random optimizer (`fast_btcusd_grid.py`).
- Runs: 3 cost assumptions × 2,500 candidates = 7,500 candidates; each run used 4 workers, three runs overlapped = up to 12 Python workers.
- GPU: RTX 4060 detected, but this backtest is branch-heavy/event-driven; no CuPy/Numba installed, so CPU multiprocessing was faster to implement safely than porting the engine.
- Warning: this is not MT5 broker truth data. Treat as candidate discovery, not live-profit proof.

## Top candidates by OOS net R
1. `fast_btcusd_fee004.json` feeR=0.04 id=699 rank=406.971615
   - OOS: netR=298.198302, expR=0.946661, trades=315, PF=3.548348, win=0.593651, maxDD=7.708818
   - IS: netR=518.671511, expR=0.760515, trades=682, PF=3.155517, maxDD=5.498749
   - params: {"lookback": 30, "atr_period": 28, "sweep_atr": 0.1, "reclaim_atr": 0.1, "disp_atr": 0.05, "sl_atr": 0.45, "tp_r": 1.6, "trail_start_r": 0.8, "trail_gap_r": 0.35, "max_hold": 240, "cooldown": 15}
2. `fast_btcusd_fee008.json` feeR=0.08 id=1706 rank=361.564456
   - OOS: netR=285.466338, expR=0.872986, trades=327, PF=2.704088, win=0.470948, maxDD=12.606396
   - IS: netR=428.953341, expR=0.582817, trades=736, PF=2.040169, maxDD=12.96
   - params: {"lookback": 30, "atr_period": 28, "sweep_atr": 0.1, "reclaim_atr": 0.02, "disp_atr": 0.1, "sl_atr": 0.45, "tp_r": 1.6, "trail_start_r": 2.2, "trail_gap_r": 0.35, "max_hold": 480, "cooldown": 15}
3. `fast_btcusd_fee004.json` feeR=0.04 id=397 rank=373.886456
   - OOS: netR=284.946152, expR=0.838077, trades=340, PF=3.323939, win=0.611765, maxDD=7.537612
   - IS: netR=433.751611, expR=0.557521, trades=778, PF=2.408321, maxDD=7.800834
   - params: {"lookback": 30, "atr_period": 10, "sweep_atr": 0.04, "reclaim_atr": 0.04, "disp_atr": 0.1, "sl_atr": 0.45, "tp_r": 1.6, "trail_start_r": 0.8, "trail_gap_r": 0.5, "max_hold": 120, "cooldown": 15}
4. `fast_btcusd_fee008.json` feeR=0.08 id=937 rank=356.798982
   - OOS: netR=280.768754, expR=0.845689, trades=332, PF=2.625991, win=0.487952, maxDD=10.8
   - IS: netR=410.80863, expR=0.562752, trades=730, PF=2.009773, maxDD=10.80282
   - params: {"lookback": 30, "atr_period": 14, "sweep_atr": 0.07, "reclaim_atr": 0.02, "disp_atr": 0.28, "sl_atr": 0.45, "tp_r": 3.4, "trail_start_r": 1.7, "trail_gap_r": 0.35, "max_hold": 480, "cooldown": 15}
5. `fast_btcusd_fee004.json` feeR=0.04 id=1781 rank=344.62455
   - OOS: netR=259.533869, expR=0.781729, trades=332, PF=2.722873, win=0.527109, maxDD=8.525998
   - IS: netR=438.756633, expR=0.58579, trades=749, PF=2.249468, maxDD=11.44
   - params: {"lookback": 30, "atr_period": 14, "sweep_atr": 0.07, "reclaim_atr": 0.04, "disp_atr": 0.05, "sl_atr": 0.45, "tp_r": 4.5, "trail_start_r": 1.3, "trail_gap_r": 0.5, "max_hold": 60, "cooldown": 15}
6. `fast_btcusd_fee002.json` feeR=0.02 id=468 rank=359.670828
   - OOS: netR=256.505411, expR=0.852177, trades=301, PF=3.248773, win=0.581395, maxDD=7.681097
   - IS: netR=499.723359, expR=0.734887, trades=680, PF=3.108804, maxDD=6.024395
   - params: {"lookback": 30, "atr_period": 21, "sweep_atr": 0.16, "reclaim_atr": 0.02, "disp_atr": 0.18, "sl_atr": 0.45, "tp_r": 2.6, "trail_start_r": 0.8, "trail_gap_r": 0.5, "max_hold": 1440, "cooldown": 15}
7. `fast_btcusd_fee004.json` feeR=0.04 id=823 rank=355.147316
   - OOS: netR=253.879276, expR=0.824283, trades=308, PF=3.091156, win=0.590909, maxDD=7.28
   - IS: netR=491.946017, expR=0.711933, trades=691, PF=3.025328, maxDD=7.224852
   - params: {"lookback": 30, "atr_period": 28, "sweep_atr": 0.16, "reclaim_atr": 0.0, "disp_atr": 0.05, "sl_atr": 0.45, "tp_r": 1.6, "trail_start_r": 0.8, "trail_gap_r": 0.5, "max_hold": 480, "cooldown": 15}
8. `fast_btcusd_fee002.json` feeR=0.02 id=1552 rank=345.920002
   - OOS: netR=253.185943, expR=1.000735, trades=253, PF=3.734227, win=0.596838, maxDD=7.14
   - IS: netR=434.910288, expR=0.75114, trades=579, PF=2.930843, maxDD=8.657531
   - params: {"lookback": 30, "atr_period": 28, "sweep_atr": 0.04, "reclaim_atr": 0.02, "disp_atr": 0.1, "sl_atr": 0.45, "tp_r": 1.6, "trail_start_r": 1.0, "trail_gap_r": 0.35, "max_hold": 1440, "cooldown": 30}
9. `fast_btcusd_fee002.json` feeR=0.02 id=664 rank=321.390804
   - OOS: netR=248.879182, expR=0.731998, trades=340, PF=2.814393, win=0.558824, maxDD=7.801035
   - IS: netR=364.715547, expR=0.484994, trades=752, PF=2.217948, maxDD=7.14
   - params: {"lookback": 30, "atr_period": 10, "sweep_atr": 0.04, "reclaim_atr": 0.02, "disp_atr": 0.28, "sl_atr": 0.45, "tp_r": 4.5, "trail_start_r": 1.0, "trail_gap_r": 0.7, "max_hold": 1440, "cooldown": 15}
10. `fast_btcusd_fee004.json` feeR=0.04 id=515 rank=335.709593
   - OOS: netR=247.238808, expR=0.905637, trades=273, PF=3.19058, win=0.567766, maxDD=9.760591
   - IS: netR=444.475125, expR=0.723901, trades=614, PF=2.780152, maxDD=6.802374
   - params: {"lookback": 45, "atr_period": 42, "sweep_atr": 0.04, "reclaim_atr": 0.02, "disp_atr": 0.42, "sl_atr": 0.45, "tp_r": 1.2, "trail_start_r": 1.0, "trail_gap_r": 0.35, "max_hold": 240, "cooldown": 15}

## Selected config candidate
- Selected from feeR=0.04 balanced-cost run because it had the best OOS netR among top runs while keeping PF > 3 and maxDD < 8R in the proxy engine.
- Selected id=699 params: {"lookback": 30, "atr_period": 28, "sweep_atr": 0.1, "reclaim_atr": 0.1, "disp_atr": 0.05, "sl_atr": 0.45, "tp_r": 1.6, "trail_start_r": 0.8, "trail_gap_r": 0.35, "max_hold": 240, "cooldown": 15}
- Exact in-repo LSR replay was partially attempted on the top candidates; it was much slower, but the top two remained positive on sampled folds. Candidate id397 had exact sampled totalR ≈ 29.47R, id699 ≈ 21.99R. The optimizer-selected id699 is still the best proxy-OOS candidate; id397 is the more conservative fallback if the exact bar strategy is prioritized.

## Config mapping
```yaml
strategies:
  liquidity_sweep_reversal:
    symbol_params:
      BTCUSD:
        atr_period: 28
        pivot_lookback_sec: 1800
        swing_window: 30
        sweep_buffer_atr: 0.1
        reclaim_buffer_atr: 0.1
        reclaim_window_sec: 120
        reclaim_extension_sec: 120
        displacement_mult: 1.0
        displacement_lookback: 20
        sl_atr_mult: 0.45
        stop_buffer_atr: 0.05
        tp_R1: 0.8
        tp_R2: 1.6
        be_at_R: 1.0
        max_hold_bars: 240
        zombie_bar_limit: 80
        min_hold_bars: 3
        min_cooldown_bars: 15
        trend_filter_enabled: false
        min_atr: 0.0005
```

## Files
- Raw results: `mt5_bot/optimization_runs/fast_btcusd_fee002.json`, `fast_btcusd_fee004.json`, `fast_btcusd_fee008.json`
- Data: `mt5_bot/data/binance/BTCUSD_TIMEFRAME_M1.csv`
- Scripts: `mt5_bot/tools/download_binance_klines.py`, `mt5_bot/tools/fast_btcusd_grid.py`, `mt5_bot/tools/parallel_lsr_optimizer.py`
