import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

# Sample Dataset
data = {
    "Height": [150, 155, 160, 165, 170, 175, 180, 185, 190, 195],
    "Weight": [50, 54, 58, 61, 65, 69, 73, 77, 82, 86]
}

# Create DataFrame
df = pd.DataFrame(data)

# Independent variable (X) and Dependent variable (y)
X = df[["Height"]]
y = df["Weight"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Plot Actual Data
plt.figure(figsize=(7, 5))

plt.scatter(
    X,
    y,
    color="blue",
    label="Actual Data"
)

# Plot Regression Line
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