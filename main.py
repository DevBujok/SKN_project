import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Można by było zrobić jescze coś takiego, ze on mowi, za ile lat wystąpi jakaś choroba z jakim prawdopodobienstwem

# Korelacja != przyczynowość

# Stwórz klasyfikatora, który przewiduje ryzyko choroby serca na podstawie danych pacjenta, takich jak wiek,
# BMI i ciśnienie. Analizuj dane oraz to jak jedna cecha wpływa na inną. Zobacz, które z cech potrzebujesz do
# predykcji, a których nie (nie zmieniają wyników modelu, są mniej istotne). Możesz użyć różnych modeli i
# porównać ich wyniki ze sobą. Przetestuj otrzymane wyniki modelu.


# Przetwarzanie wstępne danych:
#     Usunięcie duplikatów
#     Zamiana enum -> liczba - to można jakoś sprytnie zrobic po indexach unikalnych


data = pd.read_csv('heart.csv')

# potem te dane trzeba bedzie jakos odkodowac
for i in data.columns:
    # zmieniac tylko te ktore sa nieliczbowe
    if not isinstance(data[i].iloc[0], (np.float64, np.int64)):
        uniqueValues = data[i].unique() #te wartości należy zmapowac na inty
        mapDict = {}
        for j, uniqueVal in enumerate(uniqueValues):
            mapDict[uniqueVal] = j

        data[i] = data[i].map(mapDict)

X = data.drop('HeartDisease', axis=1)
Y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
