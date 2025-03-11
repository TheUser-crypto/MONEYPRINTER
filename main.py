import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def calculate_mse(y, output, n):
    e = y - output
    return (1 / n) * np.sum(e ** 2)

def robust_scale(data):

    median = np.median(data, axis=0)
    q1 = np.percentile(data, 25, axis=0)
    q3 = np.percentile(data, 75, axis=0)  
    iqr = q3 - q1

    iqr = np.where(iqr == 0, 1, iqr)

    scaled_data = (data - median) / iqr

    return scaled_data, median, iqr


def inverse_robust_scale(scaled_data, iqr, median):
    original_data = scaled_data * iqr + median
    return original_data

def prepare_data(df):

    df = df.head(7)
    X = df[['Open', 'High', 'Low', 'Volume']].values
    y = df['Close'].values.reshape(-1, 1)

    X_scaled, x_median, x_iqr = robust_scale(X)

    y_scaled, y_median, y_iqr = robust_scale(y)

    X_with_bias = np.concatenate([np.ones((X_scaled.shape[0], 1)), X_scaled], axis=1)

    return X_with_bias, y_scaled, y_median, y_iqr, x_median, x_iqr

def plot_diagram(mses, epoch):
    plt.plot(range(1, epoch + 1), mses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training Loss")
    plt.show()

def plot_actual_vs_predicted(actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual Close Price', marker='o')
    plt.plot(predicted, label='Predicted Close Price', marker='x')
    plt.xlabel("Days Ago")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Close Price")
    plt.legend()
    plt.show()

def train_model(epochs, X, y, w1, b1, w2, b2, mses, LR, n):
    for i in range(epochs):
        # Forward prop
        z1 = np.dot(X, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = z2 

        loss = calculate_mse(y, a2, n)
        mses.append(loss)

        # Backward prop
        error_output = (a2 - y)
        delta_output = error_output 

        error_hidden = delta_output.dot(w2.T)
        delta_hidden = error_hidden * sigmoid_derivative(z1)

        w2 -= (LR / n) * a1.T.dot(delta_output)
        b2 -= (LR / n) * np.sum(delta_output, axis=0, keepdims=True)
        w1 -= (LR / n) * X.T.dot(delta_hidden)
        b1 -= (LR / n) * np.sum(delta_hidden, axis=0, keepdims=True)

        if i % 1000 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    return a2, mses

def get_results(predictions, final_mse):

    week = 7
    df_first_week = df.head(week)
    actual_data = df_first_week[['Close']].values
    print("----------------------")
    print("Final MSE: ", final_mse)
    print("----------------------")
    print("Days Ago  |    Prediction  |    Actual  |    Difference  ")


    for i, item in enumerate(actual_data):
        day = i + 1
        print(f" {day} | ${predictions[i][0]} | ${item[0]} | ${predictions[i][0] - item[0]}")

def predict_tomorrow(df, w1, b1, w2, b2, x_median, x_iqr, y_median, y_iqr):
    latest_data = df[['Open', 'High', 'Low', 'Volume']].values[:1].reshape(1, -1)

    latest_data_scaled = (latest_data - x_median) / x_iqr
    latest_data_scaled = np.concatenate([np.ones((latest_data_scaled.shape[0], 1)), latest_data_scaled], axis=1)

    z1 = np.dot(latest_data_scaled, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    predicted_scaled = z2  

    predicted_price = inverse_robust_scale(predicted_scaled, y_iqr, y_median)

    return predicted_price[0][0]

input_size = 5 
hidden_size = 32
output_size = 1
np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / input_size)
b2 = np.zeros((1, output_size))

LR = 0.1
epochs = 15000

df = pd.read_csv("datasets/btc-data.csv")

X, y, y_median, y_iqr, x_median, x_iqr = prepare_data(df)
n = X.shape[0]

mses = []

prediction_close, mses = train_model(epochs, X, y, w1, b1, w2, b2, mses, LR, n)
prediction_denormalize = inverse_robust_scale(prediction_close, y_iqr, y_median)
price_tomorrow = predict_tomorrow(df, w1, b1, w2, b2, x_median, x_iqr, y_median, y_iqr)
print(f"Tomorrow price: $ {price_tomorrow}")

get_results(prediction_denormalize[:7], mses[-1])
plot_actual_vs_predicted(df['Close'].values, prediction_denormalize)