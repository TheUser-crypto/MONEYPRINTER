import requests
import time
import os
def getBTCData():
    api_key = os.getenv("API_KEY")

    base_url = f"https://api.coingecko.com/api/v3"
    
    btc_url = f"{base_url}/coins/bitcoin"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
        "x-cg-demo-api-key": api_key,
    }
    
    try:
        response = requests.get(btc_url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        btc_data = response.json()
        
        # Get BNB price for Volume BNB calculation
        bnb_url = f"{base_url}/coins/binancecoin"
        bnb_response = requests.get(bnb_url, params=params)
        bnb_response.raise_for_status()
        bnb_data = bnb_response.json()
        
        # Get OHLC data (Open, High, Low, Close) - comes in hourly intervals for past 24h
        ohlc_url = f"{base_url}/coins/bitcoin/ohlc"
        ohlc_params = {
            "vs_currency": "usd",
            "days": "1"  # Get 24 hours of data
        }
        ohlc_response = requests.get(ohlc_url, params=ohlc_params)
        ohlc_response.raise_for_status()
        ohlc_data = ohlc_response.json()
        
        # Extract current price and volume data
        current_price = btc_data["market_data"]["current_price"]["usd"]
        volume_usd = btc_data["market_data"]["total_volume"]["usd"]
        bnb_price = bnb_data["market_data"]["current_price"]["usd"]
        
        # Extract OHLC data
        # OHLC format is: [timestamp, open, high, low, close]
        # Get the 24-hour values from OHLC data
        open_price = ohlc_data[0][1]  # First entry's open price
        high_price = max(entry[2] for entry in ohlc_data)  # Maximum high
        low_price = min(entry[3] for entry in ohlc_data)   # Minimum low
        
        # Calculate volume in BTC and BNB
        volume_btc = volume_usd / current_price
        volume_bnb = volume_usd / bnb_price
        
        # Estimate tradecount (CoinGecko doesn't provide this directly)
        # Using a reasonable estimate based on BTC volume
        avg_trade_size_btc = 0.12  # Estimated average size per trade in BTC
        tradecount = int(volume_btc / avg_trade_size_btc)
        
        # Create the data point with requested keys
        data_point = {
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Volume BNB': volume_bnb,
            'Volume BTC': volume_btc,
            'tradecount': tradecount
        }
        
        return data_point
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinGecko: {e}")
        # Implement rate limit handling
        if e.response and e.response.status_code == 429:
            print("Rate limit hit, waiting 60 seconds...")
            time.sleep(60)
            return getBTCData()  # Retry after waiting
        raise