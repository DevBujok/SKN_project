import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from trained_models.model_handler import ModelHandler

# Korelacja != przyczynowość

# Stwórz klasyfikatora, który przewiduje ryzyko choroby serca na podstawie danych pacjenta, takich jak wiek,
# BMI i ciśnienie. Analizuj dane oraz to jak jedna cecha wpływa na inną. Zobacz, które z cech potrzebujesz do
# predykcji, a których nie (nie zmieniają wyników modelu, są mniej istotne). Możesz użyć różnych modeli i
# porównać ich wyniki ze sobą. Przetestuj otrzymane wyniki modelu.


# Przetwarzanie wstępne danych:
#     Usunięcie duplikatów - dane nie maja duplikatow
#     label encoder
#     usuniecie cech o niskiej korelacji


data = pd.read_csv('heart.csv')

labelEncoder = LabelEncoder()
scaler = StandardScaler()
columns_to_scale = []
# potem te dane trzeba bedzie jakos odkodowac
for i in data.columns:
    # zmieniac tylko te ktore sa nieliczbowe
    if not isinstance(data[i].iloc[0], (np.float64, np.int64)):
        uniqueValues = data[i].unique() #te wartości należy zmapowac na inty
        data[i] = labelEncoder.fit_transform(data[i])
    else:
        columns_to_scale.append(i)
columns_to_scale.remove("HeartDisease")

data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])


# dataCorrelation = data.corr()
# print(dataCorrelation.to_clipboard())
# sns.heatmap(dataCorrelation[['HeartDisease']].sort_values('HeartDisease', ascending=False), annot=True)
# plt.show()

X = data.drop(['HeartDisease', 'RestingBP', 'RestingECG'], axis=1)
Y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = classification_report(y_test, y_pred, output_dict=True)

modelHandler = ModelHandler()

modelHandler.add_model(model, "logistic_regression_prescaler", metrics, labelEncoder)
# modelHandler.delete_model("logistic_regression")
