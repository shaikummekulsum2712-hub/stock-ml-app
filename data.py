import yfinance as yf

def fetch_data(stock="^NSEI", period="1y"):
    df = yf.download(stock, period=period)
    df.dropna(inplace=True)
    return df
