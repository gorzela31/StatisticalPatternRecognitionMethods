#ZAD 1_3 Sztucznie wygenerować dane, np. dwuwymiarowe.
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

data, labels = make_blobs(n_samples=200, centers=3, n_features=2)
# Wypisanie danych
print("Wygenerowane dane:")
print(data)
print("Etykiety klas:")
print(labels)
# Wizualizacja danych
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='plasma', marker='o')
plt.title("Sztucznie wygenerowane dane dwuwymiarowe")
plt.show()