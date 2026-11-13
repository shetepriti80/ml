import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_diabetes()
# Use one feature to make the fitted curves easy to visualize
X = data.data[:, [2]]
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

degrees = [1, 2, 3, 5, 8]
train_errors, test_errors = [], []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xte = poly.transform(X_test)
    model = LinearRegression()
    model.fit(Xtr, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(Xtr)))
    test_errors.append(mean_squared_error(y_test, model.predict(Xte)))
    print(f"Degree {degree}: train MSE={train_errors[-1]:.2f}, "
          f"test MSE={test_errors[-1]:.2f}")

plt.plot(degrees, train_errors, marker="o", label="Training MSE")
plt.plot(degrees, test_errors, marker="s", label="Testing MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean Squared Error")
plt.title("Polynomial Regression: Bias-Variance Trade-off")
plt.legend()
plt.grid(True)
plt.show()