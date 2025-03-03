import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt

from api import getBTCData

fname = "datasets/btcusd_1-min_data.csv"
btc_data = getBTCData()
scaler = StandardScaler()
model = LinearRegression()

def create_prediction_diagram(x_size, y_size, test_values, pred):
    plt.figure(figsize=(x_size, y_size))
    plt.plot(test_values, label='Actual Close Prices', color='blue')
    plt.plot(pred, label='Predicted Close Prices', linestyle='--', color='red')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    plt.show()

def train_model(fname, scaler):
    df = pd.read_csv(fname)

    x = df[['High', 'Low']]
    y = df['Close']

    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, r2

def get_predicted_price(btc_data, scaler, model):

    new_data = {
        'High': [btc_data['highest']],
        'Low': [btc_data['lowest']],
    }

    new_df = pd.DataFrame(new_data)
    new_df_scaled = scaler.transform(new_df)
    predicted_price = model.predict(new_df_scaled)

    return predicted_price[0]


mae, r2 = train_model(fname, scaler)
print(f'Mean Absolute Error (MAE): {mae:.6f}')
print(f'R² Score: {r2:.4f}')

predicted_price = get_predicted_price(btc_data, scaler, model)
print("Predicted Bitcoin Price: $", predicted_price)