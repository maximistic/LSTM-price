import requests
import pandas as pd

api_key = key
ticker = 'STCK Name'    #Enter the stock you'd like to track - TSLA, AAPL
start_date = '2020-01-01'  # Specify your start date
end_date = '2023-12-31'    # Specify your end date

url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}&endDate={end_date}&token={api_key}'

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data)
df.to_csv('TSLA.csv', index=False)

print(f"Historical data for {ticker} saved to STCK.csv")
