"""Download public inputs; keep raw responses and provenance for reproducibility."""
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
TICKERS = ['SPY', 'XLE', 'XLK', 'XLU', 'XLP', 'IEF', 'XLI', 'XLF', 'XLB', 'XLV']

def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()

def main():
    DATA.mkdir(exist_ok=True)
    manifest = []
    frames = {}
    start = int(dt.datetime(2001, 1, 1, tzinfo=dt.timezone.utc).timestamp())
    end = int(dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc).timestamp())
    for ticker in TICKERS:
        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start}&period2={end}&interval=1d&events=div%2Csplits'
        raw = get(url)
        (DATA / f'{ticker}.json').write_bytes(raw)
        result = json.loads(raw)['chart']['result'][0]
        dates = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        frames[ticker] = pd.Series(result['indicators']['adjclose'][0]['adjclose'], index=dates)
        manifest.append({'ticker': ticker, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest()})
        print(ticker, len(dates), dates[-1].date(), flush=True)
    prices = pd.DataFrame(frames).sort_index()
    prices.index.name = 'date'
    prices.to_csv(DATA / 'prices.csv')
    for series in ['T5YIE', 'CPIAUCSL']:
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
        raw = get(url)
        (DATA / f'{series}.csv').write_bytes(raw)
        manifest.append({'series': series, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest()})
        print(series, 'saved', flush=True)
    (DATA / 'manifest.json').write_text(json.dumps({'downloaded_utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'sources':manifest}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
