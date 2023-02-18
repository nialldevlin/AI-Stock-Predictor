import yfinance as yf
import pandas as pd

table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]
df.to_csv('S&P500-Info.csv')
df.to_csv("S&P500-Symbols.csv", columns=['Symbol'])
sp500 = df['Symbol'].values.tolist()
data = yf.download(sp500, period='max', interval='1d', repair=True, keepna=False)
data = data['Open'].dropna(how='all')
data.to_csv("sp500_data.csv")
print(data)

