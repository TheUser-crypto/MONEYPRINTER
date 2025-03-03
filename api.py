import os
import requests

def getBTCData():
    api_key = os.getenv("API_KEY")

    url = f"https://api.freecryptoapi.com/v1/getData"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "symbol": "BTC"
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        if response:
            data = response.json()
            return data["symbols"][0]
        else:
            return None
    except Exception as e:
        print("Error", e)