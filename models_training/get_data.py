import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from trained_models.model_handler import ModelHandler

#Dane bez duplikatów, brak NAN, brak null

pd.set_option('display.expand_frame_repr', False) #linika do pokazywania pelnych statystyk bez paginacji

#Analiza data.describe():
    #Age w zakresie 28-77 - bez błędnych, mean == 50% (mediana) - rownomierny rozkład wokół środka
    #Resting BP - jest jedna wartość 0 - do usunięcia, nie ma nic pozyzej 250
    #Cholesterol - ponizej 100 bardzo rzadko spotykane - 172 wiersze z takim - zastanowic sie - wywalic wiersze czy wywalic caly cholesterol
    # FastingBS - cukier naczczo - znormalizowane do 0-1
    # MaxHR - max tetno podczas wysiłku - wartosci w normie
    # Oldpeak - o ile mV spada odcinek ST na EKG - nierealne ze na minusie a są takie wartosci

#Co mozna zrobic dla niepoprawnych danych:
    #Usunąć wiersze
    #Jak jest dużo tych danych to zastąpić na przykład srednia


#sprawdzic jeszcze outliersy tym:
#Using median calculations and IQR, outliers are identified and these data points should be removed
Q1 = df["column_name"].quantile(0.25)
Q3 = df["column_name"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[df["column_name"].between(lower_bound, upper_bound)]

def get_data():
    data = pd.read_csv('../heart.csv')
    print(data.isnull().sum())
    # hist(data[["Age"]], bins=100)
    # plt.show()

get_data()
#
# pass
# data = pd.read_csv('../heart.csv')
#
# labelEncoder = LabelEncoder()
# scaler = StandardScaler()
# columns_to_scale = []
# # potem te dane trzeba bedzie jakos odkodowac
# for i in data.columns:
#     # zmieniac tylko te ktore sa nieliczbowe
#     if not isinstance(data[i].iloc[0], (np.float64, np.int64)):
#         uniqueValues = data[i].unique() #te wartości należy zmapowac na inty
#         data[i] = labelEncoder.fit_transform(data[i])
#     else:
#         columns_to_scale.append(i)
# columns_to_scale.remove("HeartDisease")
#
# # data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
#
# # dataCorrelation = data.corr()
# # print(dataCorrelation.to_clipboard())
# # sns.heatmap(dataCorrelation[['HeartDisease']].sort_values('HeartDisease', ascending=False), annot=True)
# # plt.show()
#
# X = data.drop(['HeartDisease', 'RestingBP', 'RestingECG'], axis=1)
# Y = data['HeartDisease']
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#
# model = LogisticRegression(max_iter=10000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# metrics = classification_report(y_test, y_pred, output_dict=True)
#
# modelHandler = ModelHandler()
#
# modelHandler.add_model(model, "logistic_regression", metrics, labelEncoder)
