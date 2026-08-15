import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

# Load Dataset
iris = load_iris()

# Use two features for visualization
X = iris.data[:, [2, 3]]

# Create EM Model
gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

# Train Model
gmm.fit(X)

# Predict Clusters
clusters = gmm.predict(X)

# Probability of each cluster
probability = gmm.predict_proba(X)

# Display Cluster Means
print("Cluster Means")
print(gmm.means_)

# Display First Five Probability Values
print("\nFirst Five Probability Values")
print(probability[:5])

# Visualization
plt.figure(figsize=(7, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    cmap="viridis"
)

# Plot Cluster Centers
plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Cluster Centers"
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Expectation Maximization Clustering")
plt.legend()
plt.show()