import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from enum import Enum, auto
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    data_info_path = os.path.join(BASE_DIR, 'data_info.json')
    file_path_abs = os.path.join(BASE_DIR, filepath)
    with open(data_info_path, 'r') as f:
        data = json.load(f)

    if data['data_hash'] != hash_file(file_path_abs):
        # plik sie zmienil - update hash i return true
        data['data_hash'] = hash_file(file_path_abs)
        with open(data_info_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    else:
        #plik sie nie zmienil - return false
        return False

def if_file_exists(path: str) -> bool:
    return os.path.isfile(path)

def generate_data() -> None:
    data = pd.read_csv(os.path.join(BASE_DIR, 'heart.csv'))

    #### DATA CLEANING #####

    # RestingBP
    data = data[data["RestingBP"] != 0]  # odfiltrowac ta jedna wartosc z RestingBP

    # Cholesterol - z analizy corr i mutual_info_classif wynika ze ta cecha jest znacząca - zastąpić medianą
    cholesterol_median = data["Cholesterol"].median()
    data["Cholesterol"] = data["Cholesterol"].replace(0, cholesterol_median)  # zastąpienie 0 -> median

    #Z analizy corr te cechy mają korelacje <0.1 z HeartDisease
    data.drop(['RestingBP', 'RestingECG'], axis=1, inplace=True)

    #### DATA SCALING ####
    # robione to jest w tym miejscu bo wspolnie encodery nie pokrywaja danych ktore beda skalowane
    scaler = StandardScaler()
    columns_to_scale = []
    for i in data.columns:
        # zmieniac tylko te ktore sa nieliczbowe
        if isinstance(data[i].iloc[0], (np.float64, np.int64)):
            columns_to_scale.append(i)

    columns_to_scale.remove("HeartDisease")
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    #### DATA ENCODING ####
    data_to_encode = []  # wyodrebnienie danych do enkodowania
    for i in data.columns:
        # zmieniac tylko te ktore sa nieliczbowe
        if not isinstance(data[i].iloc[0], (np.float64, np.int64)):
            data_to_encode.append(i)

    #### LABEL_ENCODER #####
    label_encoders = {}  # Kazda cecha musi miec swoj encoder
    data_label_encoder = data.copy()  ## bez copy robi sie tylko referencja

    for i in data_to_encode:
        label_encoders[i] = LabelEncoder()
        data_label_encoder[i] = label_encoders[i].fit_transform(data_label_encoder[i])

    #### ONE_HOT ENCODER####
    # Sposob dzialania - bierze kolumny nienumeryczne i rozbija je na podcechy

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_data = one_hot_encoder.fit_transform(data[data_to_encode])  # cechy -> podcechy

    # Pobranie nazw nowych kolumn
    encoded_columns = one_hot_encoder.get_feature_names_out(data_to_encode)

    # połączenie cech i kolumn
    data_one_hot_encoder = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)

    # odfiltrowanie kolumn numerycznych z oryginału
    numerical_columns = [col for col in data.columns if col not in data_to_encode]

    # Połączenie danych
    data_one_hot_encoder = pd.concat([data[numerical_columns], data_one_hot_encoder], axis=1)

    #### ZAPIS DO PLIKU ####

    os.makedirs(os.path.join(BASE_DIR, "LABEL_ENCODER"), exist_ok=True)
    joblib.dump(label_encoders, os.path.join(BASE_DIR, "LABEL_ENCODER/label_encoders.pkl"))
    data_label_encoder.to_csv(os.path.join(BASE_DIR, "LABEL_ENCODER/heart_LABEL_ENCODER.csv"), index=False)

    os.makedirs(os.path.join(BASE_DIR, "ONE_HOT_ENCODER"), exist_ok=True)
    joblib.dump(one_hot_encoder, os.path.join(BASE_DIR, "ONE_HOT_ENCODER/one_hot_encoder.pkl"))
    data_one_hot_encoder.to_csv(os.path.join(BASE_DIR, "ONE_HOT_ENCODER/heart_ONE_HOT_ENCODER.csv"), index=False)


def get_data(encoder: EncoderEnum, force_reload: bool = False) -> tuple[pd.DataFrame, dict | OneHotEncoder]:
    # Jeśli plik się zmienił lub wymuszono ponowne przetworzenie
    if if_file_changed("heart.csv") or force_reload:
        generate_data()  # Funkcja zapisuje dane i encodery do plików

    # Wczytaj odpowiednie dane i enkoder w zależności od enkodera
    if encoder == EncoderEnum.LABEL_ENCODER:
        label_encoder_data_path = os.path.join(BASE_DIR, "LABEL_ENCODER/heart_LABEL_ENCODER.csv")
        label_encoders_path = os.path.join(BASE_DIR, "LABEL_ENCODER/label_encoders.pkl")

        # Sprawdź, czy plik danych istnieje
        if not os.path.exists(label_encoder_data_path):
            raise FileNotFoundError(
                f"Plik {label_encoder_data_path} nie istnieje. Ustaw force_reload=True, aby przetworzyć dane od nowa.")

        # Sprawdź, czy plik enkodera istnieje
        if not os.path.exists(label_encoders_path):
            raise FileNotFoundError(
                f"Plik {label_encoders_path} nie istnieje. Ustaw force_reload=True, aby przetworzyć encodery od nowa.")

        # Wczytaj dane i encodery
        data = pd.read_csv(label_encoder_data_path)
        label_encoders = joblib.load(label_encoders_path)
        return data, label_encoders

    else:  # ONE_HOT_ENCODER
        one_hot_encoder_data_path = os.path.join(BASE_DIR, "ONE_HOT_ENCODER/heart_ONE_HOT_ENCODER.csv")
        one_hot_encoders_path = os.path.join(BASE_DIR, "ONE_HOT_ENCODER/one_hot_encoder.pkl")

        # Sprawdź, czy plik danych istnieje
        if not os.path.exists(one_hot_encoder_data_path):
            raise FileNotFoundError(
                f"Plik {one_hot_encoder_data_path} nie istnieje. Ustaw force_reload=True, aby przetworzyć dane od nowa.")

        # Sprawdź, czy plik enkodera istnieje
        if not os.path.exists(one_hot_encoders_path):
            raise FileNotFoundError(
                f"Plik {one_hot_encoders_path} nie istnieje. Ustaw force_reload=True, aby przetworzyć encodery od nowa.")

        # Wczytaj dane i enkoder
        data = pd.read_csv(one_hot_encoder_data_path)
        one_hot_encoder = joblib.load(one_hot_encoders_path)
        return data, one_hot_encoder