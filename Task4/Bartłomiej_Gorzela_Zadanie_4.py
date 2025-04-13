import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

california = datasets.fetch_california_housing()
california_data = california.data
california_df = pd.DataFrame(california_data, columns=california.feature_names)

# Wizualiacja danych za pomocą histogramów
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(california_df['MedInc'], bins=100, color='skyblue', edgecolor='black')
plt.title(f'Histogram: MedInc')
plt.xlabel('MedInc [$10,000]')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)
plt.hist(california_df['HouseAge'], bins=100, color='salmon', edgecolor='black')
plt.title(f'Histogram: HouseAge')
plt.xlabel('HouseAge [years]')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Wizualizacja 2D oryginalnych danych
plt.figure(figsize=(6, 5))
plt.scatter(california_df['MedInc'], california_df['HouseAge'], color='green', edgecolor='k')
plt.xlabel('MedInc [$10,000]')
plt.ylabel('HouseAge [years]')
plt.title('Wizualizacja 2D oryginalnych danych')
plt.show()

# DBSCAN
attributes_for_scaling = california_df[['MedInc', 'HouseAge']]
scaler = StandardScaler()
attributes_scaled = scaler.fit_transform(attributes_for_scaling)
dbscan = DBSCAN(eps=0.3, min_samples=10)
# dbscan = DBSCAN(eps=0.1, min_samples=10)
# dbscan = DBSCAN(eps=0.3, min_samples=15)
# dbscan = DBSCAN(eps=0.1, min_samples=15)

labels = dbscan.fit_predict(attributes_scaled)
california_df['dbscan_label'] = labels

# Wizualizacja 2D z etykietami DBSCAN
plt.figure(figsize=(6, 5))
colors = ['red' if label == -1 else 'blue' for label in labels]
plt.scatter(california_df['MedInc'], california_df['HouseAge'], c=colors, edgecolor='k')
plt.xlabel('MedInc [$10,000]')
plt.ylabel('HouseAge [years]')
plt.title('Wizualizacja 2D dla eps=0.3 i min_samples=10')
# plt.title('Wizualizacja 2D dla eps=0.1 i min_samples=10')
# plt.title('Wizualizacja 2D dla eps=0.3 i min_samples=15')
# plt.title('Wizualizacja 2D dla eps=0.1 i min_samples=15')
plt.show()