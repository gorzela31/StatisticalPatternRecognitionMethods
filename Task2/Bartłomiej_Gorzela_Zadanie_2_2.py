#ZAD 2_2 Wizualizacja wybranych atrybutów przy użyciu histogramu.
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

# Wczytanie zbioru danych Iris
iris = datasets.load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)

# Histogram dla "sepal length (cm)" - Iris Dataset
plt.hist(iris_df['sepal length (cm)'], bins = len(iris_df['sepal length (cm)']))
plt.title('Histogram: Sepal Length (cm) - Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Histogram dla "sepal width (cm)" - Iris Dataset
plt.hist(iris_df['sepal width (cm)'], bins = len(iris_df['sepal width (cm)']))
plt.title('Histogram: Sepal Width (cm) - Iris Dataset')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid()
plt.show()