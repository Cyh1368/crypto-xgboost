# Real Data Acquisition Guide — `Crypto-XGBoost`

## Problem Summary

The current `backtester_v2` dataset (`BTC_USDT_50k.parquet`) contains **two fabricated data columns** that corrupt ~12 of the 47 model features:

| Column | Status | Impact |
|---|---|---|
| `open`, `high`, `low`, `close`, `volume` | ✅ Real (Coinbase) | Features are reliable |
| `bids` / `asks` (orderbook) | ❌ **Proxy** — one real snapshot scaled to every bar | `obi_tau*`, `spread_bps`, `depth_ratio_*`, `book_pressure_*` are near-constant; zero predictive signal |
| `funding_rate` | ❌ **Hardcoded** `0.0001` | `funding_rate`, `funding_8h_ma` carry zero signal |

Additionally, the high correlation (~0.44) reported in `backtester_v1` is **not a real signal** — it is an artifact of cross-ticker data leakage caused by a global index split across coins covering the same time window. The model learned contemporaneous cross-sectional correlations, not temporal ones.

---

## Data Requirements

To train an uncontaminated model with all 47 features carrying genuine signal, you need three data streams fetched for **each ticker**, aligned to the **same 15-minute bar timestamps**:

1. **OHLCV candles** — already available; continue using the same source
2. **L2 orderbook snapshots** — one snapshot per 15-minute bar (or higher resolution, then sampled)
3. **Funding rates** — one rate per 8-hour funding interval, forward-filled to 15-minute bars

---

## Source Options by Data Type

### 1. OHLCV Candles (already solved — no changes needed)

Continue using the Coinbase Advanced Trade REST API or any equivalent CEX. For futures/perps trading, prefer the **perps instrument** (e.g., `BTC-PERP`) so OHLCV and funding are from the same product.

---

### 2. L2 Orderbook — Real Historical Snapshots

Historical L2 data is not available from free REST endpoints. Your options in order of cost:

#### Option A — Tardis.dev (too expensive, not using this)
Tardis stores tick-by-tick L2 snapshots for all major perp exchanges going back several years.

```
https://tardis.dev
```

- **Coverage**: Binance, Bybit, OKX, dYdX, and 40+ others
- **Format**: CSV or Parquet, per-exchange, per-symbol, per-day
- **Cost**: ~$50–200/month depending on symbols and history depth; free tier available for recent 1-day samples
- **Resolution**: Full depth, millisecond timestamps
- **How to use**:
  1. Download raw L2 snapshot files for each ticker and date range
  2. Filter to the timestamp closest to each 15-minute bar close (e.g., `bar_close - 5s`)
  3. Store the top N bid/ask levels (10–20 levels is sufficient for the current feature set)

```python
# Tardis Python client
pip install tardis-dev

from tardis_dev import datasets
datasets.download(
    exchange="binance-futures",
    data_types=["book_snapshot_25"],   # top-25 levels
    from_date="2024-01-01",
    to_date="2024-12-31",
    symbols=["BTCUSDT", "ETHUSDT"],
    api_key="YOUR_API_KEY",
)
```

#### Option B — Binance Data Portal (Use this: Free, limited depth)
Binance publishes daily orderbook snapshot files for its futures market.

```
https://data.binance.vision/?prefix=data/futures/um/daily/bookDepth/
```

- **Coverage**: Binance USDT-M futures only
- **Format**: CSV (`.zip`), one file per symbol per day
- **Cost**: Free
- **Limitation**: Snapshots are taken once per second; you must select the row nearest each 15-minute bar

```bash
# Example download for BTCUSDT, one day
wget "https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-01-01.zip"
```

```python
import pandas as pd

def load_binance_book_snapshot(zip_path: str, bar_timestamps: pd.DatetimeIndex, levels: int = 10):
    df = pd.read_csv(zip_path, compression='zip', names=[
        'timestamp', 'side', 'price', 'quantity', 'update_id'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp')

    snapshots = {}
    for ts in bar_timestamps:
        # Grab the snapshot closest to (but not after) each bar close
        window = df[df['timestamp'] <= ts].tail(200)
        bids = window[window['side'] == 'b'][['price', 'quantity']].head(levels).values.tolist()
        asks = window[window['side'] == 'a'][['price', 'quantity']].head(levels).values.tolist()
        snapshots[ts] = {'bids': bids, 'asks': asks}
    return snapshots
```

#### Option C — WebSocket Recording (Free, forward-only)
If you are willing to start fresh from today, subscribe to the exchange L2 WebSocket stream and record one snapshot per 15-minute bar.

- **Cost**: Free
- **Limitation**: You only accumulate history going forward; no historical backfill

```python
# Binance Futures partial depth stream (top 20 levels, 250ms updates)
# wss://fstream.binance.com/stream?streams=btcusdt@depth20@250ms

import asyncio, websockets, json, pandas as pd

async def record_orderbook(symbol="btcusdt", levels=20):
    url = f"wss://fstream.binance.com/stream?streams={symbol}@depth{levels}@250ms"
    async with websockets.connect(url) as ws:
        while True:
            msg = json.loads(await ws.recv())
            snapshot = msg['data']
            ts = pd.Timestamp.now(tz='UTC').floor('15min')
            # persist snapshot keyed by bar timestamp
            save_snapshot(ts, snapshot['bids'][:levels], snapshot['asks'][:levels])
```

---

### 3. Funding Rates — Real Historical Data

Funding rates are paid every 8 hours on perpetual futures. All major exchanges expose a REST endpoint for historical funding.

#### Binance Futures — Funding Rate History

```
GET https://fapi.binance.com/fapi/v1/fundingRate
```

Parameters:
- `symbol`: e.g. `BTCUSDT`
- `startTime` / `endTime`: Unix milliseconds
- `limit`: max 1000 per call

```python
import requests, pandas as pd

def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.Series:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(url, params={
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000
        })
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]['fundingTime'] + 1

    df = pd.DataFrame(rows)
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['fundingRate'] = df['fundingRate'].astype(float)
    df = df.set_index('fundingTime').sort_index()

    # Forward-fill to 15-minute bars
    bar_index = pd.date_range(df.index[0], df.index[-1], freq='15min', tz='UTC')
    return df['fundingRate'].reindex(bar_index).ffill()
```

#### Bybit Funding Rate History (alternative)

```
GET https://api.bybit.com/v5/market/funding/history
```

Parameters: `category=linear`, `symbol`, `startTime`, `endTime`, `limit` (max 200)

---

## Assembling the Final Parquet

Once all three streams are collected, merge them into a single parquet that matches the schema expected by `feature_engineering.py`:

```python
import pandas as pd
import numpy as np

def build_real_dataset(
    ohlcv: pd.DataFrame,         # columns: open, high, low, close, volume; DatetimeIndex UTC 15min
    snapshots: dict,             # {timestamp: {'bids': [[price,qty],...], 'asks': [[price,qty],...]}}
    funding: pd.Series,          # DatetimeIndex UTC, funding rate per bar
    levels: int = 10
) -> pd.DataFrame:

    df = ohlcv.copy()
    df['bids'] = [
        np.array(snapshots.get(ts, {}).get('bids', []))
        for ts in df.index
    ]
    df['asks'] = [
        np.array(snapshots.get(ts, {}).get('asks', []))
        for ts in df.index
    ]
    df['funding_rate'] = funding.reindex(df.index).ffill()

    # Drop bars with missing orderbook data
    df = df[df['bids'].apply(lambda x: len(x) > 0)]

    df.to_parquet(f"backtester_v2/data/raw/{symbol}_real.parquet")
    return df
```

The resulting parquet will have **identical column names and dtypes** to the existing proxy file, so no changes to `feature_engineering.py` are required.

---

## Training Split — Preventing Leakage (Critical)

When training on multiple tickers, **never use a global index split**. Always split by time:

```python
TRAIN_END   = pd.Timestamp("2024-09-30 23:59:00", tz="UTC")
TEST_END    = pd.Timestamp("2024-11-30 23:59:00", tz="UTC")
# Everything after TEST_END is validation

X_train = X[X.index <= TRAIN_END]
X_test  = X[(X.index > TRAIN_END) & (X.index <= TEST_END)]
X_val   = X[X.index > TEST_END]
```

This ensures **no ticker's future data leaks into another ticker's training window**, regardless of the order rows are concatenated.

---

## Checklist Before Re-training

- [ ] Orderbook snapshots fetched from a real exchange for every bar (not scaled from a single snapshot)
- [ ] Funding rates fetched from the exchange REST API and forward-filled to 15-minute bars
- [ ] Train / test / val split is strictly **time-based**, not index-based
- [ ] Validation set is chronologically **after** all training data across every ticker
- [ ] `obi_tau1` unique value count >> 3 (sanity check that orderbook data is real)
- [ ] `funding_rate` has more than 1 unique value across the dataset
