import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# --------------------------
# Load Dataset
# --------------------------

iris = load_iris()

X = iris.data
y = iris.target


# --------------------------
# Train-Test Split
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------
# Feature Scaling
# --------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------
# KNN
# --------------------------

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_scaled, y_train)

knn_pred = knn.predict(X_test_scaled)


# --------------------------
# Logistic Regression
# --------------------------

lr = LogisticRegression(random_state=42)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)


# --------------------------
# Decision Tree
# --------------------------

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)


# --------------------------
# Accuracy Scores
# --------------------------

results = pd.DataFrame({
    "Algorithm": [
        "KNN",
        "Logistic Regression",
        "Decision Tree"
    ],
    
    "Accuracy": [
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, dt_pred)
    ]
})


# --------------------------
# Display Results
# --------------------------

print(results)