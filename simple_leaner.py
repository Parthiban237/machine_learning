import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Sample Dataset
data = {
    "Height": [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
    "Weight": [50, 54, 58, 61, 65, 69, 73, 77, 82, 86]
}

df = pd.DataFrame(data)

# Input and Output
X = df[["Height"]]
y = df["Weight"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Create Model
model = LinearRegression()


# Train the Model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("RMSE:")
print(rmse)

print("\nR2 Score:")
print(r2)


# Regression Equation
print("\nRegression Equation:")
print(
    f"Weight = {model.coef_[0]:.2f} * Height + "
    f"{model.intercept_:.2f}"
)


# Plot
plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y,
    color="blue",
    label="Actual Data"
)

plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Simple Linear Regression")

plt.legend()
plt.grid(True)
plt.show()