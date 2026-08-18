import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load Dataset
iris = load_iris()

# Use two features for visualization
# Column 2 = Petal Length
# Column 3 = Petal Width
X = iris.data[:, [2, 3]]


# ------------------------
# Elbow Method
# ------------------------

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    wcss.append(model.inertia_)


# Plot Elbow Method
plt.figure(figsize=(7, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.show()


# ------------------------
# K-Means Model
# ------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Predict clusters
clusters = kmeans.fit_predict(X)


# ------------------------
# Cluster Centers
# ------------------------

print("Cluster Centers:")
print(kmeans.cluster_centers_)


# ------------------------
# Visualization
# ------------------------

plt.figure(figsize=(7, 5))

# Plot data points
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=clusters,
    cmap="viridis"
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering")

plt.legend()

plt.show()