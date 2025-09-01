# scripts/fetch_full_data.py
import requests
import pandas as pd
from pathlib import Path
import time

# API_KEY = "1LBPCJCSQWU7YAKN"  # Replace with UserProfile.api_key if needed
BASE_URL = "https://www.alphavantage.co/query"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

symbols = ["TCS.BSE", "RELIANCE.BSE", "HDFCBANK.BSE", "BHARTIARTL.BSE", "ICICIBANK.BSE"]

def get_full_daily(symbol):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",   # ✅ fetch complete history
        "apikey": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise Exception(f"Error fetching {symbol}: {data.get('Note', data)}")

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume",
        "7. dividend amount": "Dividend",
        "8. split coefficient": "SplitCoeff"
    })

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()  # oldest → newest
    df = df.astype(float)

    # Save per stock
    file_path = DATA_DIR / f"{symbol.replace('.', '_')}_full.csv"
    df.to_csv(file_path, index=True)
    print(f"✅ Saved {symbol} → {file_path}")
    return df

if __name__ == "__main__":
    for symbol in symbols:
        try:
            df = get_full_daily(symbol)
            print(f"{symbol}: {len(df)} rows")
            # ⏳ Respect Alpha Vantage free-tier rate limit (5 calls per min)
            time.sleep(15)
        except Exception as e:
            print(f"❌ Failed for {symbol}: {e}")
