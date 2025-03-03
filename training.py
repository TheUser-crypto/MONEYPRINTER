import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Not needed but for info :D
def calculate_mse(y, y_hoc):
    e = y - y_hoc
    return 0.5 * np.sum(e ** 2)

# Number of students
n = 100

# Number of iterations
epochs = 1000

df = pd.read_csv("datasets/sleep_casus.csv")

l_step = 0.01

X = df[['hours_studied', 'hours_slept']].values
y = df['score'].values.reshape(-1, 1)

w = np.array([[10], [5]])
y_hoc = np.dot(X, w)

mses = []

for i in range(epochs):
    y_hoc = np.dot(X, w)
    mse = calculate_mse(y, y_hoc)
    mses.append(float(mse))
    e = y - y_hoc
    g = -(1 / n) * np.dot(X.T, e) 
    w = w - l_step * g

print(mses)

print("New Weight: ", w)


# Generate plot
plt.plot(range(1, epochs + 1), mses, marker='o', markersize=3, linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Cost Reduction over Epochs")
plt.grid(True)
plt.show()