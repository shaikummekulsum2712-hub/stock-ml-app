import yfinance as yf
import pandas as pd

def fetch_data(symbol):
    try:
        df = yf.download(symbol, period="1y", interval="1d")

        if df.empty:
            return None

        return df

    except Exception as e:
        return None
