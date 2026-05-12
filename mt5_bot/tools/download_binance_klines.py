from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _fetch(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[Any]:
    params = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
    )
    url = f"https://api.binance.com/api/v3/klines?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = resp.read()
    return json.loads(payload.decode("utf-8"))


def download(symbol: str, interval: str, days: int, output: Path, sleep_sec: float = 0.08) -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=int(days))
    start_ms = _ms(start)
    end_ms = _ms(end)
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = _fetch(symbol=symbol, interval=interval, start_ms=cursor, end_ms=end_ms)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + 60_000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(rows) % 10000 == 0:
            print(f"downloaded={len(rows)} cursor={datetime.fromtimestamp(cursor/1000, tz=timezone.utc).isoformat()}", flush=True)
        time.sleep(float(sleep_sec))
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    dedup = []
    for row in rows:
        key = int(row[0])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    dedup.sort(key=lambda item: int(item[0]))
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "volume"])
        for row in dedup:
            ts = datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc).isoformat()
            writer.writerow([ts, row[1], row[2], row[3], row[4], row[5]])
    print(f"saved={output} rows={len(dedup)} start={start.isoformat()} end={end.isoformat()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--output", default="mt5_bot/data/binance/BTCUSD_TIMEFRAME_M1.csv")
    args = ap.parse_args()
    download(args.symbol, args.interval, args.days, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
