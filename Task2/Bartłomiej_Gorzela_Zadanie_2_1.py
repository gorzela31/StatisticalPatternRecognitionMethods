#ZAD 2_1 Dla dwóch wybranych zbiorów danych ich wizualizacja przy pomocy wykresów 2D i 3D.
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import pandas as pd

iris = datasets.load_iris()
wine = datasets.load_wine()

iris_data = iris.data
iris_target = iris.target
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)
iris_df['Class'] = iris_target

wine_data = wine.data[:, :3]
wine_target = wine.target
wine_df = pd.DataFrame(wine_data, columns=["Alcohol", "Malic Acid", "Ash"])
wine_df['Class'] = wine_target

# Wykresy 2D
plt.figure(figsize=(8, 6))
sns.scatterplot(x=iris_df['sepal length (cm)'], y=iris_df['sepal width (cm)'], hue=iris_df['Class'], palette="viridis")
plt.title("Wykres 2D: Iris Dataset")
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Class')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=wine_df['Alcohol'], y=wine_df['Malic Acid'], hue=wine_df['Class'], palette='viridis')
plt.title("Wykres 2D: Wine Dataset")
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.legend(title='Class')
plt.show()

# Wykresy 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(iris_df['sepal length (cm)'], iris_df['sepal width (cm)'], iris_df['petal length (cm)'], c=iris_df['Class'], cmap='viridis')
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Sepal Width (cm)')
ax.set_zlabel('Petal Length (cm)')
plt.title("Wykres 3D: Iris Dataset")
plt.legend(*scatter.legend_elements(), title="Class")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(wine_df['Alcohol'], wine_df['Malic Acid'], wine_df['Ash'], c=wine_df['Class'], cmap='viridis')
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Ash')
plt.title("Wykres 3D: Wine Dataset")
plt.legend(*scatter.legend_elements(), title="Class")
plt.show()