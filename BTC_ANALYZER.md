# BTC Up/Down analyzer

Double-click **Start BTC Analyzer.cmd**. Select 5-minute or 15-minute BTC contracts.

If dependencies are missing, run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analyzer.txt
```

UP and DOWN each show the best bid (sell), best ask (buy), spread, quote age,
and number of received quote events. Prices are contract cents: 60¢ means
0.60 per share. These are Polymarket outcome prices, not the BTC/USD spot price.

Market discovery uses the public Gamma API. Quotes come directly from the
public CLOB WebSocket, using book snapshots, price changes and best-bid/ask
events. No wallet or API key is needed. This application does not place orders.
It reconnects automatically and discovers the next contract at interval end.
The GUI samples the latest in-memory quote every 25 ms; this is a display target,
not a guarantee of end-to-end latency. Event delay includes clock offset between
your computer and the exchange. Quote age turns amber after 10 seconds without
a quote event; a quiet market can still have a healthy connection. Prices are
hidden on disconnect and expiry until the current connection supplies quotes.

```powershell
.\.venv\Scripts\python.exe btc_analyzer.py --minutes 15
.\.venv\Scripts\python.exe btc_analyzer.py --smoke
.\.venv\Scripts\python.exe -m pytest test_btc_analyzer.py -q
```

Protocol: https://docs.polymarket.com/market-data/realtime-data
