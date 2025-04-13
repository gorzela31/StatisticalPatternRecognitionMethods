from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Wczytanie zbioru digits (cyfry 0–9)
digits = load_digits()
digits_data, digits_target = digits.data, digits.target

#Podział na zbiór treningowy i testowy
digits_data_train, digits_data_test, digits_target_train, digits_target_test = train_test_split(digits_data, digits_target, test_size=0.2, random_state=50)

#Standaryzacja cech
scaler = StandardScaler()
digits_data_train_scaled = scaler.fit_transform(digits_data_train)
digits_data_test_scaled = scaler.transform(digits_data_test)

#Zadanie 5.1
print("Zadanie 5.1: Użycie klasyfikatora kNN z różnymi miarami odległości i różną liczbą sąsiadów:")
amount_of_neighbours = [1, 3, 5, 7]
metrics = ['euclidean','chebyshev', 'manhattan']
print("-------------------------------------------------------------------------")
for i in amount_of_neighbours:
    for metric in metrics:
        kNN = KNeighborsClassifier(n_neighbors = i, metric = metric)
        kNN.fit(digits_data_train_scaled, digits_target_train)
        y_predicted = kNN.predict(digits_data_test_scaled)
        acc = accuracy_score(digits_target_test, y_predicted)
        print(f"k = {i}, metric = {metric} -> Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print("-------------------------------------------------------------------------")

#Zadanie 5.2
print("\nZadanie 5.2: Użycie klasyfikatora SVM dla różnych parametrów jądra:")
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
print("-------------------------------------------------------------------------")
for kernel in kernels:
    SVM = SVC(kernel = kernel)
    SVM.fit(digits_data_train_scaled, digits_target_train)
    y_predicted = SVM.predict(digits_data_test_scaled)
    acc = accuracy_score(digits_target_test, y_predicted)
    print(f"Kernel = {kernel} -> Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print("-------------------------------------------------------------------------")