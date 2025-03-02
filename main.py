import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

from gecko import getBTCData


def create_prediction_diagram(x_size, y_size, test_values, pred):
    plt.figure(figsize=(x_size, y_size))
    plt.plot(test_values, label='Actual Close Prices', color='blue')
    plt.plot(pred, label='Predicted Close Prices', linestyle='--', color='red')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    plt.show()


fname = "BTC_ds.csv"

df = pd.read_csv(fname)

x = df[['Open', 'High', 'Low', 'Volume BNB', 'Volume BTC', 'tradecount']]
y = df['Close']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.6f}')
print(f'R² Score: {r2:.4f}')


btc_data = getBTCData()

new_data = {
    'Open': [btc_data['Open']],
    'High': [btc_data['High']],
    'Low': [btc_data['Low']],
    'Volume BNB': [btc_data['Volume BNB']],
    'Volume BTC': [btc_data['Volume BTC']],
    'tradecount': [btc_data['tradecount']]
}

new_df = pd.DataFrame(new_data)
new_df_scaled = scaler.transform(new_df)
predicted_price = model.predict(new_df_scaled)

print("Btc Data:", btc_data)
print("Predicted Bitcoin Price: $", predicted_price[0])

percentage_error = abs((btc_data['Open'] - predicted_price) / btc_data['Open']) * 100
print("Percentage Error: ", percentage_error[0])