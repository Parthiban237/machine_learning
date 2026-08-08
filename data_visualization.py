import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Step 2: Load Iris Dataset
iris = load_iris()
# Step 3: Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
# Add Species Name
df["Species"] = [
    iris.target_names[i]
    for i in iris.target
]
print(df.head())
# Step 4: Pair Plot
sns.pairplot(df, hue="Species")
plt.show()
# Step 5: Histogram with KDE
plt.figure(figsize=(7,5))
sns.histplot(
    data=df,
    x="sepal length (cm)",
    kde=True,
    bins=10
)
plt.title("Histogram of Sepal Length")
plt.show()
# Step 6: Correlation Heatmap
plt.figure(figsize=(6,5))
corr = df.drop("Species", axis=1).corr()
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show() 
