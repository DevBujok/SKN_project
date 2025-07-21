import json
import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import hist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from enum import Enum, auto
import hashlib

from trained_models.model_handler import ModelHandler

#Dane bez duplikatów, brak NAN, brak null

pd.set_option('display.expand_frame_repr', False) #linika do pokazywania pelnych statystyk bez paginacji

#Co mozna zrobic dla niepoprawnych danych:
    #Usunąć wiersze
    #Jak jest dużo tych danych to zastąpić na przykład srednia

# Age: age of the patient [years] - zakres 28-27, bez błędnych, mean == 50% (mediana) - rownomierny rozkład wokół środka
# Sex: sex of the patient [M: Male, F: Female]
# ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# RestingBP: resting blood pressure [mm Hg] - jest jedna wartość 0 - do usunięcia, nie ma nic pozyzej 250
# Cholesterol: serum cholesterol [mm/dl] -  ponizej 100 bardzo rzadko spotykane - 172 wiersze z zerowym cholesterolem - zastanowic sie - wywalic wiersze czy wywalic caly cholesterol
# FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# MaxHR: maximum heart rate achieved [Numeric value between 60 and 202] - wartosci w normie
# ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# Oldpeak: oldpeak = ST [Numeric value measured in depression] - występuja ujemne wartosci i one czasami sa spotykane - chatGPT
# ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# HeartDisease: output class [1: heart disease, 0: Normal]

def get_mutual_info_classif(data):
    """
    Przyjmuje enkodowane dane!!! Pokazuje, czy dana cecha ma wartość predykcyjną
    """
    X = data.drop(columns=["HeartDisease"])  # lub inna nazwa zmiennej celu
    y = data["HeartDisease"]

    importances = mutual_info_classif(X.fillna(0), y)
    ranking = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print(ranking)

class EncoderEnum(Enum):
    LABEL_ENCODER = auto()
    ONE_HOT_ENCODER = auto()

def hash_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def if_file_changed(filepath):
    with open('data_info.json', 'r') as f:
        data = json.load(f)

    if data['data_hash'] != hash_file(filepath):
        # plik sie zmienil - update hash i return true
        data['data_hash'] = hash_file(filepath)
        with open('data_info.json', 'w') as f:
            json.dump(data, f, indent=2)
        return True
    else:
        #plik sie nie zmienil - return false
        return False

def if_file_exists(path: str) -> bool:
    return os.path.isfile(path)

def get_data(encoder: EncoderEnum, force_reload: bool = False):
    #dodac if: jezeli plik sie zmienil albo nie istnieje przetworzony plik
    #uproszczenie - założenie, że pliki istnieja bo za bardzo sie rozrasta
    #uproszczenie - za kazdym razem jak plik sie zmieni to one_hot i label robione od nowa
    if if_file_changed("heart.csv") or force_reload: #jezeli sie plik zmienil
        data = pd.read_csv('heart.csv')

        #### DATA CLEANING #####

        # RestingBP
        data = data[data["RestingBP"] != 0]  # odfiltrowac ta jedna wartosc z RestingBP

        # Cholesterol - z analizy corr i mutual_info_classif wynika ze ta cecha jest znacząca - zastąpić medianą
        cholesterol_median = data["Cholesterol"].median()
        data["Cholesterol"] = data["Cholesterol"].replace(0, cholesterol_median)  # zastąpienie 0 -> median

        #### DATA SCALING ####
        #robione to jest w tym miejscu bo wspolnie encodery nie pokrywaja danych ktore beda skalowane
        scaler = StandardScaler()
        columns_to_scale = []
        for i in data.columns:
            # zmieniac tylko te ktore sa nieliczbowe
            if isinstance(data[i].iloc[0], (np.float64, np.int64)):
                columns_to_scale.append(i)

        columns_to_scale.remove("HeartDisease")
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

        #### LABEL_ENCODER #####

        label_encoders = {} #Kazda cecha musi miec swoj encoder
        data_label_encoder = data.copy() ## bez copy robi sie tylko referencja

        for i in data_label_encoder.columns:
            # zmieniac tylko te ktore sa nieliczbowe
            if not isinstance(data_label_encoder[i].iloc[0], (np.float64, np.int64)):
                label_encoders[i] = LabelEncoder()
                data_label_encoder[i] = label_encoders[i].fit_transform(data_label_encoder[i])
            else:
                columns_to_scale.append(i)
        columns_to_scale.remove("HeartDisease")


    else:
        #plik sie nie zmienil - zwroc odpowieni
        pass



get_data(encoder=EncoderEnum.LABEL_ENCODER, force_reload=True)

#

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
