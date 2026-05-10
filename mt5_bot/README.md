

# MT5 Quant Bot (Reliability Rebuild)

Professional-grade architecture with:

- MT5 heartbeat keep-alive with auto-reconnect on IPC timeout
- Emergency shutdown handler (close all open positions on exit)
- Unified `live` and `backtest` runtime modes
- External signal injection (`json_file` or TCP socket JSON lines)
- Telegram alerts for system/trade/error events
- State-machine strategy engine (`mean_reversion_sm`, `vol_breakout_sm`)

## Install

```bash
pip install MetaTrader5 PyYAML pandas
```

## Run (Live)

```bash
cd mt5_bot
python runner.py --config config.yaml --mode live
```

## Run (Backtest)

```bash
python runner.py --config config.yaml --mode backtest
```

## Watchdog Health Check (lock / process / desired-state)

```bash
python watchdog_healthcheck.py
python watchdog_healthcheck.py --verbose
python watchdog_healthcheck.py --fix --notify --notify-cooldown-sec 900
```

`--fix` will attempt only targeted recovery for this failure class:
- remove stale/malformed lock files
- remove duplicate watchdog/runner processes beyond one instance each

The script reports:
- `BLOCK` issues (requires manual or `--fix` action before restart)
- `WARN` issues (recommendation only)

?êÎèô ?êÍ? Î∞∞Ïπò(Í∂åÏû•):

```bash
cd mt5_bot
python watchdog_healthcheck.py
python watchdog_healthcheck.py --fix --notify --notify-cooldown-sec 900
```

Îß?1Î∂??êÎèô ?§Ìñâ ?±Î°ù(Í¥ÄÎ¶¨Ïûê Í∂åÌïú CMD Í∂åÏû•):

```bash
register_watchdog_healthcheck_scheduler.cmd
```

?±Î°ù ?¥Ï†ú:

```bash
register_watchdog_healthcheck_scheduler.cmd /delete
```

## One-cycle smoke check

## Watchdog Auto Check

```bash
cd mt5_bot
watchdog_healthcheck.cmd
watchdog_healthcheck.cmd --fix
```

Register 1-minute scheduler:

```bash
register_watchdog_healthcheck_scheduler.cmd
```

Remove scheduler:

```bash
register_watchdog_healthcheck_scheduler.cmd /delete
```

```bash
python runner.py --config config.yaml --once
```

## Historical Data Format (Backtest)

Place CSV files in `backtest.data_dir` (default: `./data`) as:

- `SYMBOL_TIMEFRAME.csv` (preferred), example: `BTCUSD_TIMEFRAME_M5.csv`
- or `SYMBOL.csv`

Required columns:

- `open`, `high`, `low`, `close`

Optional columns:

- `time` or `timestamp` (ISO timestamp or unix seconds)
- `tick_volume` or `volume`

## External Signal JSON Example

File mode (`external_signals.source: json_file`):

```json
[
  {
    "id": "sig-1001",
    "symbol": "BTCUSD",
    "action": "BUY",
    "reason": "LLM breakout conviction",
    "confidence": 0.82,
    "ttl_seconds": 120
  }
]
```

Socket mode (`external_signals.source: socket`) expects JSON-lines over TCP:

```text
{"id":"sig-2001","symbol":"XAUUSD","action":"EXIT","reason":"risk_off"}\n
```

## Project Structure

```text
mt5_bot/
  runner.py
  config.yaml
  core/         # config, lifecycle, runtime, domain models
  brokers/      # MT5 live gateway + backtest gateway
  strategies/   # state-machine strategies
  signals/      # JSON/socket external signal adapters
  execution/    # order manager
  alerts/       # Telegram notifier
  storage/      # JSON state/event persistence
  data/         # historical data loader
  utils/        # indicators
```
