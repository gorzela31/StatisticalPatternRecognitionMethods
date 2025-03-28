#ZAD 1_1	Wypisać dane zaczytane z repozytorium
from sklearn.datasets import load_iris
import pandas as pd

# Wczytanie danych Iris z sklearn
iris = load_iris()
# Konwersja do DataFrame dla czytelności
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['class'] = iris.target
print("Dane zaczytane z repozytorium:")
print(data)