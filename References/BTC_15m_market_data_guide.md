# BTC 15-Minute Prediction Market: Data Collection Guide

This guide covers API access, settlement data sourcing, and backtesting data collection for
trading a BTC up/down XGBoost classifier on 15-minute binary prediction markets.

---

## 2. Kalshi: Platform Overview

Kalshi's BTC 15-minute markets ask: *"Will Bitcoin be up or down in the next 15 minutes?"*
Each market opens at the start of a 15-minute window and resolves at its close. Contracts are
binary: YES pays $1.00, NO pays $0.00.

**Series ticker:** `KXBTC15M`  
**Example market ticker:** `KXBTC15M-26MAY110900` (May 11, 2026, 09:00 window)

**Settlement logic:** At expiry, Kalshi collects 60 consecutive per-second BRTI readings
(the final 60 seconds before close) and averages them. If the average is ≥ the opening BRTI
snapshot, YES resolves to $1.00. The top and bottom 20% of the 60 readings are trimmed before
averaging on some markets (trimmed mean). The opening reference is also a BRTI reading taken
at window open.

---

## 3. Kalshi API Reference

### 3.1 Authentication

Market data (read-only) requires **no authentication**. Trading endpoints require an API key.

```
Base URL: https://external-api.kalshi.com/trade-api/v2
Sandbox URL: https://demo-api.kalshi.co/trade-api/v2
```

**API key auth** (for order placement):  
Include in every authenticated request:
```
Authorization: Token <your_api_key>
```
Keys are created in your Kalshi account dashboard under Settings → API Keys.  
Rate limits: 10 req/s for trading endpoints; 100 req/s for data endpoints.

**Python SDK:**
```bash
pip install kalshi-python
```

---

### 3.2 Key Endpoints

#### Discover the KXBTC15M Series

```
GET /trade-api/v2/series/KXBTC15M
```

Returns series metadata including frequency, category, and description.

```python
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
resp = requests.get(f"{BASE}/series/KXBTC15M")
print(resp.json())
```

---

#### List All Markets in the Series

```
GET /trade-api/v2/markets
```

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `series_ticker` | string | `KXBTC15M` |
| `status` | string | `open`, `closed`, `settled`, `all` |
| `min_close_ts` | int | Unix timestamp lower bound |
| `max_close_ts` | int | Unix timestamp upper bound |
| `limit` | int | Max results per page (default 100, max 1000) |
| `cursor` | string | Pagination cursor from previous response |

```python
def get_all_btc15m_markets(status="all"):
    markets = []
    cursor = None
    while True:
        params = {"series_ticker": "KXBTC15M", "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/markets", params=params).json()
        markets.extend(resp["markets"])
        cursor = resp.get("cursor")
        if not cursor:
            break
    return markets
```

Each market object contains:
- `ticker` — unique ID, e.g. `KXBTC15M-26MAY110900`
- `open_time` / `close_time` — Unix timestamps for the 15-minute window
- `yes_bid_dollars` / `yes_ask_dollars` — current best quotes
- `last_price_dollars` — last traded price (implied probability)
- `volume_fp` — total volume in contracts
- `status` — `open`, `closed`, or `settled`
- `result` — `yes` or `no` after settlement

---

#### Get the Orderbook for a Live Market

```
GET /trade-api/v2/markets/{ticker}/orderbook
```

Returns real-time YES and NO bids at all price levels.

```python
ticker = "KXBTC15M-26MAY110900"
resp = requests.get(f"{BASE}/markets/{ticker}/orderbook").json()
# resp["orderbook_fp"]["yes_dollars"] → list of [price, size] pairs
# resp["orderbook_fp"]["no_dollars"]  → list of [price, size] pairs
```

---

#### Get Live Market Candlesticks (for currently open/recent markets)

```
GET /trade-api/v2/markets/{ticker}/candlesticks
```

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start_ts` | int | Unix timestamp (inclusive) |
| `end_ts` | int | Unix timestamp (inclusive) |
| `period_interval` | int | Candle size in minutes: `1`, `60`, or `1440` |

This is the primary endpoint for intra-window odds data. Use `period_interval=1` for
minute-by-minute odds throughout each 15-minute window.

```python
import time

ticker = "KXBTC15M-26MAY110900"
# Example: fetch 1-minute candles for the full window
window_open  = 1746961200  # replace with actual Unix timestamp
window_close = 1746962100  # 15 minutes later

resp = requests.get(
    f"{BASE}/markets/{ticker}/candlesticks",
    params={"start_ts": window_open, "end_ts": window_close, "period_interval": 1}
).json()

for candle in resp["candlesticks"]:
    print(candle["end_period_ts"],
          candle["yes_bid"]["close"],
          candle["yes_ask"]["close"],
          candle["price"]["close"],
          candle["volume"])
```

Each candlestick contains:
- `end_period_ts` — end of the candle period (Unix timestamp)
- `yes_bid` — open/high/low/close of the YES bid price
- `yes_ask` — open/high/low/close of the YES ask price
- `price` — open/high/low/close/mean of last-traded price
- `volume` — contracts traded in this candle
- `open_interest` — open interest at end of candle

---

#### Get Historical Market Candlesticks (for archived markets)

Markets older than Kalshi's historical cutoff are moved to a separate archive. Use this endpoint
for all markets that have already been settled and archived.

```
GET /trade-api/v2/historical/markets/{ticker}/candlesticks
```

Same parameters as the live candlesticks endpoint above. Valid `period_interval` values:
`1` (1 minute), `60` (1 hour), `1440` (1 day).

```python
ticker = "KXBTC15M-25DEC240730"  # an archived historical market
resp = requests.get(
    f"{BASE}/historical/markets/{ticker}/candlesticks",
    params={"start_ts": 0, "end_ts": int(time.time()), "period_interval": 1}
).json()
```

---

#### Get Historical Markets (for bulk ticker discovery)

To get all historical settled markets, use:

```
GET /trade-api/v2/historical/markets
```

**Query parameters:**

| Parameter | Type | Description |
|---|---|---|
| `series_ticker` | string | `KXBTC15M` |
| `limit` | int | Max per page |
| `cursor` | string | Pagination cursor |

```python
def get_historical_btc15m_markets():
    markets = []
    cursor = None
    while True:
        params = {"series_ticker": "KXBTC15M", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/historical/markets", params=params).json()
        markets.extend(resp["markets"])
        cursor = resp.get("cursor")
        if not cursor:
            break
    return markets
```

---

#### Get Historical Cutoff Timestamp

To determine which markets are "live" vs. "archived", check:

```
GET /trade-api/v2/historical/cutoff_timestamps
```

Markets with `close_time` before the cutoff require the `/historical/` endpoints.

---

#### Place an Order (requires API key)

```
POST /trade-api/v2/orders
```

```python
import requests

headers = {"Authorization": f"Token {API_KEY}", "Content-Type": "application/json"}
payload = {
    "ticker": "KXBTC15M-26MAY110900",
    "side": "yes",           # "yes" or "no"
    "action": "buy",         # "buy" or "sell"
    "count": 10,             # number of contracts
    "type": "limit",         # "limit" or "market"
    "yes_price": 55,         # price in cents (1–99)
    "expiration_ts": int(time.time()) + 60,  # optional: order TTL
}
resp = requests.post(f"{BASE}/orders", json=payload, headers=headers).json()
```

---

#### WebSocket: Real-Time Odds Stream

Use the WebSocket API to subscribe to live orderbook and trade updates during an open window.
This is essential for intra-window signal generation at sub-minute granularity.

```
WSS: wss://trading-api.kalshi.com/trade-api/ws/v2
```

Authentication is required for the WebSocket. Subscribe to a market's orderbook after connecting:

```python
import asyncio, websockets, json

async def stream_orderbook(api_key, ticker):
    uri = "wss://trading-api.kalshi.com/trade-api/ws/v2"
    async with websockets.connect(uri, extra_headers={"Authorization": f"Token {api_key}"}) as ws:
        # Subscribe to orderbook deltas
        await ws.send(json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_delta"], "market_tickers": [ticker]}
        }))
        async for message in ws:
            data = json.loads(message)
            print(data)

asyncio.run(stream_orderbook(API_KEY, "KXBTC15M-26MAY110900"))
```

WebSocket channels relevant to this strategy:

| Channel | Description |
|---|---|
| `orderbook_delta` | Incremental orderbook updates (best for real-time odds) |
| `ticker` | Top-of-book quote updates |
| `trade` | Every individual trade as it executes |

---

## 4. Settlement Data: CF Benchmarks BRTI

### What BRTI Is

The **CME CF Bitcoin Real-Time Index (BRTI)** is a per-second Bitcoin benchmark price
aggregated from multiple regulated spot exchanges: Bitstamp, Coinbase, Gemini, itBit (Paxos),
Kraken, LMAX Digital, Bullish, and Crypto.com. It is the official settlement source for all
Kalshi BTC markets.

**Kalshi settlement procedure:**
1. At window open (e.g., 09:00:00 ET), the BRTI reading is captured as the **opening reference**.
2. In the 60 seconds before close (e.g., 09:14:00–09:14:59), 60 per-second BRTI readings
   are collected.
3. The **trimmed mean** (excluding the top/bottom 20% = 12 readings each) of those 60 values
   is the **settlement price**.
4. If settlement price ≥ opening reference → YES wins; otherwise NO wins.

### Accessing BRTI Data

**CF Benchmarks API** (licensed, requires contacting CF Benchmarks):
```
Base URL: https://www.cfbenchmarks.com/api/v1/
Auth: Basic <Base64(username:key)>
Index ticker: BRTI
```

Relevant endpoints:
```
GET /api/v1/values?instruments=BRTI        # Last 1 hour of per-second values
GET /api/v1/history?instruments=BRTI&...   # Historical values (licensed)
```
Historical BRTI data requires a commercial license. Contact:
`licensing@cfbenchmarks.com`

**WebSocket (real-time, also licensed):**
```
WSS: wss://www.cfbenchmarks.com/ws/v1/
Subscribe: {"type": "subscribe", "instruments": ["BRTI"]}
```

### Practical Alternative: Use Constituent Exchange Data

Since BRTI is not freely available historically, use data from a **constituent exchange** as
a high-fidelity proxy. **Coinbase Advanced Trade** or **Kraken** are the recommended sources
because they are named BRTI constituent exchanges.

**Important caveats:**
- Any single exchange's price can diverge from BRTI by a few cents/dollars during
  illiquid periods or exchange-specific events.
- For your XGBoost feature set (which includes order book microstructure and VWAP), using
  a constituent exchange's data is appropriate and standard practice.
- For final settlement validation, always use the official BRTI value from Kalshi's market
  result, not your exchange proxy.

**Coinbase Advanced Trade REST API (free, no auth for public endpoints):**
```
Base URL: https://api.coinbase.com/api/v3/brokerage/
Spot ticker: BTC-USD
```

```python
# Coinbase: historical candles (granularity in seconds)
import requests

resp = requests.get(
    "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles",
    params={
        "start": "1746961200",   # Unix timestamp
        "end": "1746962100",
        "granularity": "ONE_MINUTE"  # ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, etc.
    }
).json()
# resp["candles"] → list of {start, low, high, open, close, volume}
```

For per-second data (to replicate the 60-second settlement average), use the Coinbase
WebSocket product channel:
```
WSS: wss://advanced-trade-ws.coinbase.com
Subscribe: {"type":"subscribe","product_ids":["BTC-USD"],"channel":"ticker"}
```
This emits a ticker update on every trade, not exactly every second. As an alternative,
poll the REST `GET /api/v3/brokerage/market/products/BTC-USD/ticker` endpoint once per second
and record prices during the settlement window.

**Kraken REST API (free):**
```python
resp = requests.get(
    "https://api.kraken.com/0/public/OHLC",
    params={"pair": "XBTUSD", "interval": 1}  # 1-minute bars
).json()
```

---

## 5. Backtesting Data Collection

To backtest a mid-window entry strategy, you need **two datasets per market window**:

1. **Odds data** — the YES price (implied probability) over time within each 15-minute window,
   ideally by the minute.
2. **BTC price data** — BRTI (or proxy) per-second data to reconstruct settlement outcomes
   and build XGBoost features.

### 5.1 Collecting Historical Kalshi Odds Data

The strategy requires odds by the **minute** within each 15-minute window.
Kalshi's 1-minute candlesticks provide exactly this.

#### Step 1: Enumerate all historical KXBTC15M market tickers

```python
import requests, time

BASE = "https://external-api.kalshi.com/trade-api/v2"

def get_all_kxbtc15m_tickers():
    """Fetch all KXBTC15M market tickers (live + archived)."""
    all_markets = []

    # 1. Live markets (open, closed, settled but not yet archived)
    cursor = None
    while True:
        params = {"series_ticker": "KXBTC15M", "status": "all", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/markets", params=params).json()
        all_markets.extend(resp.get("markets", []))
        cursor = resp.get("cursor")
        if not cursor:
            break

    # 2. Archived historical markets
    cursor = None
    while True:
        params = {"series_ticker": "KXBTC15M", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/historical/markets", params=params).json()
        all_markets.extend(resp.get("markets", []))
        cursor = resp.get("cursor")
        if not cursor:
            break

    return all_markets
```

#### Step 2: Fetch 1-minute candlesticks for each market

Each KXBTC15M market spans exactly 15 minutes, so there will be at most 15 candles per market
at 1-minute resolution.

```python
import time

def fetch_candlesticks(ticker, open_ts, close_ts, is_historical=False):
    """Returns list of 1-minute candles for a given market window."""
    endpoint = (
        f"{BASE}/historical/markets/{ticker}/candlesticks"
        if is_historical
        else f"{BASE}/markets/{ticker}/candlesticks"
    )
    resp = requests.get(
        endpoint,
        params={"start_ts": open_ts, "end_ts": close_ts, "period_interval": 1}
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("candlesticks", [])
```

#### Step 3: Determine which endpoint to use per market

```python
def get_historical_cutoff():
    resp = requests.get(f"{BASE}/historical/cutoff_timestamps").json()
    return resp["cutoff_timestamps"]["market"]  # Unix timestamp

cutoff = get_historical_cutoff()

def is_archived(market):
    return market["close_time"] < cutoff
```

#### Step 4: Full bulk collection loop

```python
import pandas as pd
import time

def collect_all_odds(output_path="btc15m_odds.parquet", delay=0.1):
    markets = get_all_kxbtc15m_tickers()
    cutoff  = get_historical_cutoff()
    rows = []

    for m in markets:
        ticker    = m["ticker"]
        open_ts   = m["open_time"]    # may be string ISO or int — normalize as needed
        close_ts  = m["close_time"]
        result    = m.get("result")   # "yes", "no", or None if not settled

        archived  = m["close_time"] < cutoff
        candles   = fetch_candlesticks(ticker, open_ts, close_ts, is_historical=archived)

        for c in candles:
            rows.append({
                "ticker":           ticker,
                "window_open_ts":   open_ts,
                "window_close_ts":  close_ts,
                "candle_end_ts":    c["end_period_ts"],
                "yes_bid_close":    float(c["yes_bid"]["close"]),
                "yes_ask_close":    float(c["yes_ask"]["close"]),
                "price_close":      float(c["price"]["close"]),
                "price_mean":       float(c["price"]["mean"]),
                "volume":           float(c["volume"]),
                "open_interest":    float(c["open_interest"]),
                "result":           result,
            })

        time.sleep(delay)  # respect rate limits

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    return df
```

**Output schema** (one row per candle per market window):

| Column | Description |
|---|---|
| `ticker` | Market ticker, e.g. `KXBTC15M-26MAY110900` |
| `window_open_ts` | Unix timestamp of window open |
| `window_close_ts` | Unix timestamp of window close |
| `candle_end_ts` | End of this 1-minute candle |
| `yes_bid_close` | YES bid price at candle close (0.01–0.99) |
| `yes_ask_close` | YES ask price at candle close |
| `price_close` | Last traded price at candle close |
| `price_mean` | Mean traded price during the candle |
| `volume` | Contracts traded in this candle |
| `open_interest` | Open interest at candle end |
| `result` | Final settlement: `"yes"` or `"no"` |

> **Note on minute resolution:** Kalshi's API offers 1-minute as the finest candlestick
> granularity. For finer resolution (sub-minute/per-trade), use `GET /trade-api/v2/markets/{ticker}/trades`
> (live) or `GET /trade-api/v2/historical/trades` (historical). These return every individual
> trade with a timestamp, price, and size, allowing full tick reconstruction.

#### 5.1a Per-Trade Historical Data (highest granularity)

```
GET /trade-api/v2/historical/trades
```

| Parameter | Description |
|---|---|
| `ticker` | e.g. `KXBTC15M-26MAY110900` |
| `limit` | Max 1000 per page |
| `cursor` | Pagination cursor |
| `min_ts` | Start Unix timestamp |
| `max_ts` | End Unix timestamp |

```python
def fetch_all_trades(ticker):
    trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE}/historical/trades", params=params).json()
        trades.extend(resp.get("trades", []))
        cursor = resp.get("cursor")
        if not cursor:
            break
    return trades
```

Each trade record contains: `created_time`, `ticker`, `price`, `count` (contracts), `taker_side`.

---

### 5.2 Collecting Historical BTC Price Data (for feature engineering)

Your XGBoost model uses 47 features including order book microstructure, momentum, and funding
rates. These are constructed from the underlying BTC price data — not from the Kalshi market
itself. The following sources match (or closely approximate) BRTI constituent exchange data.

#### Recommended: Coinbase Advanced Trade (BTC-USD spot)

Coinbase is a named BRTI constituent exchange. Its spot BTC-USD data is a high-quality proxy.

```python
# Hourly / minute OHLCV (public, no auth)
resp = requests.get(
    "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles",
    params={"start": str(start_ts), "end": str(end_ts), "granularity": "ONE_MINUTE"}
).json()
```

Granularity options: `ONE_MINUTE`, `FIVE_MINUTE`, `FIFTEEN_MINUTE`, `THIRTY_MINUTE`,
`ONE_HOUR`, `TWO_HOUR`, `SIX_HOUR`, `ONE_DAY`.

For your feature set (which includes lags at 1, 3, 6, 12, 48 bars on 15-minute bars),
you should collect at least **6 months of 1-minute or 15-minute BTC-USD OHLCV data**.

Limit per call: 350 candles. For bulk historical collection:

```python
def fetch_coinbase_candles(start_ts, end_ts, granularity="ONE_MINUTE"):
    """Fetch Coinbase BTC-USD candles in paginated 350-candle chunks."""
    all_candles = []
    chunk_seconds = 350 * 60  # 350 minutes per call at 1-min resolution
    t = start_ts
    while t < end_ts:
        chunk_end = min(t + chunk_seconds, end_ts)
        resp = requests.get(
            "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/candles",
            params={"start": str(t), "end": str(chunk_end), "granularity": granularity}
        ).json()
        all_candles.extend(resp.get("candles", []))
        t = chunk_end
        time.sleep(0.2)
    return all_candles
```

#### Alternative: Binance BTC/USDT (not a BRTI constituent, but widely used)

```python
resp = requests.get(
    "https://api.binance.com/api/v3/klines",
    params={
        "symbol": "BTCUSDT",
        "interval": "1m",
        "startTime": start_ts * 1000,   # Binance uses milliseconds
        "endTime":   end_ts   * 1000,
        "limit": 1000
    }
).json()
# Returns: [open_time, open, high, low, close, volume, close_time, ...]
```

Binance provides 1-minute data going back several years but is **NOT a BRTI constituent
exchange**. Use it for general feature engineering but do not use it to validate settlement.

#### Orderbook Data (for OBI / Depth features)

Your feature set includes Order Book Imbalance computed at multiple decay factors. Live orderbook
snapshots can be collected from Coinbase or Kraken. Historical orderbook snapshots are not
freely available at full depth — options:

- **Tardis.dev** — provides tick-by-tick historical orderbook data for BTC across multiple
  exchanges (subscription required). This is the industry standard for institutional backtesting.
  `https://tardis.dev`
- **Kaiko** — high-quality historical crypto market data including order book snapshots
  (enterprise pricing). `https://www.kaiko.com`
- **Crypto Lake** — more affordable historical crypto data for retail quants.
  `https://crypto-lake.com`

For a free approximation, use **Binance level-2 snapshot history** via their Data Vision portal:
`https://data.binance.vision/?prefix=data/spot/daily/depth/BTCUSDT/`

---

### 5.3 Collecting Funding Rate Data (Feature: Funding Rate)

Your feature set includes current and 8-hour MA funding rate. Funding rates are perpetual
futures-specific. Collect from Binance (most liquid perps market):

```python
# Current funding rate
resp = requests.get(
    "https://fapi.binance.com/fapi/v1/fundingRate",
    params={"symbol": "BTCUSDT", "limit": 1000}
).json()
# Historical: same endpoint with startTime and endTime
```

Funding rates are published every 8 hours (00:00, 08:00, 16:00 UTC).

---

## 6. Data Alignment for Backtesting

The critical alignment requirement: for each Kalshi market window, the XGBoost model is run
**at decision time** (e.g., T+3 minutes into the window) to predict whether BTC will finish
up or down by T+15 minutes. The label is the Kalshi settlement result (YES/NO).

Recommended data schema for a backtesting row:

| Field | Source | Description |
|---|---|---|
| `window_open_ts` | Kalshi market | Window start Unix timestamp |
| `window_close_ts` | Kalshi market | Window close Unix timestamp |
| `decision_ts` | Derived | Time at which the model fires (e.g., open_ts + 180s) |
| `btc_open_price` | Coinbase 1m candle | BTC price at window open (proxy for BRTI opening ref) |
| `btc_decision_price` | Coinbase 1m candle | BTC price at decision time |
| `[47 XGBoost features]` | Coinbase, Binance, Kalshi | Feature vector at decision_ts |
| `yes_mid_at_decision` | Kalshi 1m candle | `(yes_bid + yes_ask) / 2` at decision_ts |
| `result` | Kalshi settlement | `1` = YES (up), `0` = NO (down) |
| `btc_settlement_price` | BRTI / Coinbase proxy | Average BRTI in last 60 seconds of window |

---

## 7. Requirements Summary

### Python Dependencies

```bash
pip install requests pandas pyarrow websockets asyncio
pip install kalshi-python          # official Kalshi Python SDK
pip install py-clob-client         # Polymarket CLOB SDK (if using Polymarket)
```

### Kalshi Account Setup

1. Create an account at `https://kalshi.com`
2. Complete KYC verification
3. Generate an API key in Settings → API Keys (needed only for trading, not data)
4. Fund account (minimum $1)

### API Key Permissions Required

- **Market data only:** No API key needed
- **Live trading:** API key with trading permissions

---

## 8. Supplementary: Polymarket Details

Documented here for completeness. **Not recommended for US users** due to geoblocking.

### Settlement

Polymarket BTC 15-minute markets resolve via the **Chainlink BTC/USD data stream** on
Polygon. The opening price is the Chainlink feed value at window open; the closing price is
the Chainlink feed value at window close. If `close ≥ open`, UP wins.

To retrieve Chainlink on-chain price history for settlement validation, query the
Chainlink BTC/USD Price Feed contract on Polygon:
```
Contract: 0xc907E116054Ad103354f2D350FD2514433D57F6F (Polygon mainnet)
Method: getRoundData(roundId)
```

### API Endpoints

```
Gamma (market discovery): https://gamma-api.polymarket.com
CLOB (orderbook/trading): https://clob.polymarket.com
```

**Find a BTC 15m market by slug:**
```python
slug = f"btc-updown-15m-{window_close_timestamp}"  # e.g. btc-updown-15m-1746962100
resp = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"slug": slug}
).json()
token_id = resp[0]["clobTokenIds"][0]  # YES token ID
```

**Get historical odds (1-minute resolution via CLOB):**
```python
resp = requests.get(
    "https://clob.polymarket.com/prices-history",
    params={"market": token_id, "interval": "max", "fidelity": 1}  # fidelity in minutes
).json()
# resp["history"] → list of {"t": unix_ts, "p": price}
```

**Note:** The Polymarket CLOB `prices-history` endpoint returns sampled midpoint prices,
not full OHLCV candlesticks. Fidelity controls the time resolution (1 = 1-minute intervals).
For full tick data, query the on-chain event logs via The Graph:
```
GraphQL: https://api.thegraph.com/subgraphs/name/polymarket/matic-markets
```