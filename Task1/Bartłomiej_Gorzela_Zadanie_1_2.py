#ZAD 1_2 Wypisać dane pobrane z pliku .csv.
import pandas as pd
data = pd.read_csv('Task1/iris.csv')
print("Dane zaczytane z pliku .csv o dataset 'Iris':")
print(data)