#ZAD 3
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import pandas as pd

iris = datasets.load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(iris_data, columns=iris.feature_names)

# 3.1 Podać zakres zmienności wartości atrybutów.
range = iris_df.max() - iris_df.min()
print("Zakres zmienności wartości atrybutów w zbiorze Iris:")
print(range)

# 3.2 Obliczyć średnią dla wartości atrybutów wybranego zbioru danych.
mean_values = iris_df.mean()
print("Średnie wartości atrybutów w zbiorze Iris:")
print(mean_values)

# 3.3 Obliczyć odchylenie standardowe wartości atrybutów.
std_values = iris_df.std()
print("Odchylenia standardowe wartości atrybutów w zbiorze Iris:")
print(std_values)

# 3.4 Przeskalować do pewnego przedziału. Wybraną metodą zwizualizować różnice między tak otrzymanymi wartościami cech a tymi przed przeskalowaniem.
scaler = MinMaxScaler(feature_range=(0, 1))
iris_scaled = scaler.fit_transform(iris_df)
iris_scaled_df = pd.DataFrame(iris_scaled, columns=iris.feature_names)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
for i, feature in enumerate(iris.feature_names):
    # Przed przeskalowaniem
    axes[0, i].hist(iris_df[feature], bins=len(iris_df[feature]), color='blue', label=f'Before Scaling')
    axes[0, i].set_title(f'Before Scaling - {feature}')
    axes[0, i].set_xlabel(feature)
    axes[0, i].set_ylabel('Frequency')
    # Po przeskalowaniu
    axes[1, i].hist(iris_scaled_df[feature], bins=len(iris_scaled_df[feature]), color='green', label=f'After Scaling')
    axes[1, i].set_title(f'After Scaling [0, 1] - {feature}')
    axes[1, i].set_xlabel(feature)
    axes[1, i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3.5 Wykonać standaryzację wartości atrybutów. Wybraną metodą zwizualizować różnice.
scaler = StandardScaler()
iris_standardized = scaler.fit_transform(iris_df)
iris_standardized_df = pd.DataFrame(iris_standardized, columns=iris.feature_names)

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
for i, feature in enumerate(iris.feature_names):
    # Przed standaryzacją
    axes[0, i].hist(iris_df[feature], bins=len(iris_df[feature]), color='blue', label=f'Before Standardization')
    axes[0, i].set_title(f'Before Standardization - {feature}')
    axes[0, i].set_xlabel(feature)
    axes[0, i].set_ylabel('Frequency')
    # Po standaryzacji
    axes[1, i].hist(iris_standardized_df[feature], bins=len(iris_standardized_df[feature]), color='green', label=f'After Standardization')
    axes[1, i].set_title(f'After Standardization - {feature}')
    axes[1, i].set_xlabel(feature)
    axes[1, i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3.6 Przeprowadzić normalizację wartości atrybutów (przy użyciu różnych norm). Wybraną metodą zwizualizować różnice.
normalizer_L1 = Normalizer(norm='l1')  # Norma L1
normalizer_L2 = Normalizer(norm='l2')  # Norma L2
iris_L1_normalized = normalizer_L1.fit_transform(iris_df)
iris_L2_normalized = normalizer_L2.fit_transform(iris_df)
iris_L1_df = pd.DataFrame(iris_L1_normalized, columns=iris.feature_names)
iris_L2_df = pd.DataFrame(iris_L2_normalized, columns=iris.feature_names)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i, feature in enumerate(iris.feature_names):
    # Przed normalizacją
    axes[0, i].hist(iris_df[feature], bins=len(iris_df[feature]), color='blue', label=f'Before Normalization')
    axes[0, i].set_title(f'Before Normalization - {feature}')
    axes[0, i].set_xlabel(feature)
    axes[0, i].set_ylabel('Frequency')
    # Norma L1
    axes[1, i].hist(iris_L1_df[feature], bins=len(iris_L1_df[feature]), color='green', label=f'L1 Normalization')
    axes[1, i].set_title(f'L1 Normalization - {feature}')
    axes[1, i].set_xlabel(feature)
    axes[1, i].set_ylabel('Frequency')
    # Norma L2
    axes[2, i].hist(iris_L2_df[feature], bins=len(iris_L2_df[feature]), color='orange', label=f'L2 Normalization')
    axes[2, i].set_title(f'L2 Normalization - {feature}')
    axes[2, i].set_xlabel(feature)
    axes[2, i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
print("Wartości atrybutów zbioru Iris przed normalizacją:") 
print(iris_df)
print("Wartości atrybutów zbioru Iris po normalizacji normą L1:")
print(iris_L1_df)
print("Wartości atrybutów zbioru Iris po normalizacji normą L2:")
print(iris_L2_df)

# 3.7 Oczyścić dane (uzupełnić/usunąć brakujące wartości, usunąć duplikaty)
iris_df.loc[0:3, 'sepal length (cm)'] = None
print("Dane przed oczyszczaniem:")
print(iris_df)
print("Ilość brakujących wartości każdej z cech:")
print(iris_df.isnull().sum())

# Uzupełnianie brakujących wartości poprzez średnią danego atrybutu
iris_df['sepal length (cm)'].fillna(iris_df['sepal length (cm)'].mean(), inplace=True)
print("Dane po oczyszczaniu:")
print(iris_df)
print("Ilość brakujących wartości każdej z cech:")
print(iris_df.isnull().sum())

# Usuwanie wierszy z brakującymi wartościami
iris_df.loc[0:3, 'sepal width (cm)'] = None
print("Dane przed oczyszczaniem:")
print(iris_df)
print("Ilość brakujących wartości każdej z cech:")
print(iris_df.isnull().sum())

iris_df.dropna(inplace=True)
print("Dane po oczyszczaniu:")
print(iris_df)
print("Ilość brakujących wartości każdej z cech:")
print(iris_df.isnull().sum())

# Usuwanie duplikatów
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df = pd.concat([iris_df, iris_df])
print("Dane zduplikowane:")
print(iris_df)
print("Liczba duplikatów przed usunięciem:")
print(iris_df.duplicated().sum()) 
iris_df.drop_duplicates(inplace=True)
print("Dane po usunieciu duplikatów:")
print(iris_df)
print("Liczba duplikatów po usunięciu:")
print(iris_df.duplicated().sum()) 