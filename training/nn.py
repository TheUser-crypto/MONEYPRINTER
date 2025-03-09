import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def calculate_mse(y, y_hat, n):
    e = y - y_hat
    return (1 / n) * np.sum(e ** 2)

def prepare_data(df):
    X = df[['x1', 'x2']].values
    y = df['y'].values.reshape(-1, 1)

    # Standardization
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    bias_column = np.ones((X_normalized.shape[0], 1))
    X_with_bias = np.concatenate([bias_column, X_normalized], axis=1)

    return X_with_bias, y

def gradient_descent(epoch, X, y, w_input, w_output, bias_hidden, bias_output, n, l_step):
    mses = []
    for i in range(epoch):

        ## FORWARD PROP

        # Input naar verborgen laag
        y_hat = np.dot(X, w_input) + bias_hidden
        output = sigmoid(y_hat)

        # Verborgen laag naar outputlaag
        y_hat_input = np.dot(output, w_output) + bias_output
        y_hat_output = sigmoid(y_hat_input)

        # Fout berekenen
        e = y - y_hat_output

        ## BACKWARD PROP

        output_delta = e * sigmoid_derivative(y_hat_output)

        hidden_delta = output_delta.dot(w_output.T) * sigmoid_derivative(output)

        # Gewichten en bias bijwerken
        w_output += output.T.dot(output_delta) * l_step
        bias_output += np.sum(output_delta, axis=0, keepdims=True) * l_step

        w_input += X.T.dot(hidden_delta) * l_step
        bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * l_step

        mse = calculate_mse(y, y_hat_output, n)
        mses.append(float(mse))

    return y_hat_output, mses

def plot_diagram(mses):
    plt.plot(range(1, epoch + 1), mses, marker='o', markersize=3, linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Reduction over Epochs")
    plt.grid(True)
    plt.show()


df = pd.read_csv("datasets/XOR_dataset.csv")

X, y = prepare_data(df)
n = X.shape[0]
epoch = 10000
np.random.seed(42)

# Gewichten en biases initialiseren
w_input = np.random.randn(X.shape[1], 4)  # X.shape[1] -> 3 (bias + 2 inputs), 4 hidden neurons
w_output = np.random.randn(4, 1)  # 4 hidden neurons -> 1 output neuron
bias_hidden = np.random.randn(1, 4)  # Bias voor verborgen laag
bias_output = np.random.randn(1, 1)  # Bias voor outputlaag
l_step = 0.1

# Gradient Descent uitvoeren
y_hat_output, mses = gradient_descent(epoch, X, y, w_input, w_output, bias_hidden, bias_output, n, l_step)

print(mses)

# Resultaat tonen
print("Output na training: ", y_hat_output)

print("Uitslagen van NN:")
for i in range(4):
    print("1" if y_hat_output[i] > 0.5 else "0")



plot_diagram(mses)