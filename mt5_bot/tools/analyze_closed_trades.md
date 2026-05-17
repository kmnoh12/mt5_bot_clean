# Closed Trade Postmortem CLI

Read-only analyzer for bot-closed trades. It parses `events.jsonl`, reconstructs filled entry and broker close context, writes a chart/report pair, and appends a compact learning sample.

Manual run:

```bash
python tools/analyze_closed_trades.py --events events.jsonl --output-dir reports/trade_postmortems --symbol BTCUSD --limit 10 --mt5
```

Safe fallback without MT5:

```bash
python tools/analyze_closed_trades.py --events events.jsonl --output-dir reports/trade_postmortems --symbol BTCUSD --limit 10 --no-mt5
```

Important properties:

- Does not place, close, modify, or cancel orders.
- Does not change live config or strategy thresholds.
- Uses `postmortem_index.jsonl` for idempotence; pass `--force` only to rebuild a known report.
- Backs up an existing output file before overwriting it.
- If `MetaTrader5` or `matplotlib` is unavailable, it still writes a structured JSON/Markdown report and a lightweight fallback PNG.

Outputs:

- `reports/trade_postmortems/<trade_key>.json`
- `reports/trade_postmortems/<trade_key>.md`
- `reports/trade_postmortems/assets/<trade_key>.png`
- `reports/trade_postmortems/postmortem_index.jsonl`
- `reports/trade_postmortems/learning_samples.jsonl`
- `reports/trade_postmortems/vision_prompt.md`
