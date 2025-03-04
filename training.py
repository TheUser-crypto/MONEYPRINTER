import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Not needed but for info :D
def calculate_mse(y, y_hat):
    e = y - y_hat
    return 0.5 * np.sum(e ** 2)


def prepare_data(df):
    X = df[['hours_studied', 'hours_slept']].values
    y = df['score'].values.reshape(-1, 1)

    # Standardization
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    bias_column = np.ones((X_normalized.shape[0], 1))
    X_with_bias = np.concatenate([bias_column, X_normalized], axis=1)

    return X_with_bias, y


def gradient_descent(epochs, X_with_bias, y, n):
    mses = []
    l_step = 0.01
    w = np.random.randn(X.shape[1], 1)

    for i in range(epochs):
        y_hat = np.dot(X_with_bias, w)
        mse = calculate_mse(y, y_hat)
        mses.append(float(mse))
        e = y - y_hat
        g = -(1 / n) * np.dot(X_with_bias.T, e) 
        w = w - l_step * g
    return w, mses

def plot_diagram(mses):
    plt.plot(range(1, epochs + 1), mses, marker='o', markersize=3, linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Reduction over Epochs")
    plt.grid(True)
    plt.show()

epochs = 1000
df = pd.read_csv("datasets/sleep_casus.csv")

X, y = prepare_data(df)
n = X.shape[0]
new_w, mses = gradient_descent(epochs, X, y, n)

print(mses)
plot_diagram(mses)
