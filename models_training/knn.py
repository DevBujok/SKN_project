import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from trained_models.model_handler import ModelHandler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('../heart.csv')

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

#knn wrazliwe na skalowanie
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

X = data.drop(['HeartDisease', 'RestingBP', 'RestingECG'], axis=1)
Y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = classification_report(y_test, y_pred, output_dict=True)

modelHandler = ModelHandler()

modelHandler.add_model(model, "knn_5", metrics, labelEncoder, scaler)
