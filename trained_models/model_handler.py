#Klasa pomocnicza do zapisywania modeli
#przyjmuje: nazwa modelu, model
#działanie - na podstawie przesłanych danych tworzy folder i dodaje tam pkl i dane o wytrenowanym modelu do jsona
import json, joblib, os, shutil
from pathlib import Path

from data_handler.get_data import EncoderEnum

#sciezki robione jako absolutne bo na roznych systemach roznie te sciezki sie wywoluja
modelsFolder = Path(__file__).resolve().parent
modelsInfoURL = modelsFolder / "models.json"
print(modelsInfoURL)

class ModelHandler:

    @staticmethod
    def _get_models_from_json():
        """
        Prywatna statyczna metoda do pobierania listy modeli z jsona
        """
        try: #proba otwarcia pliku i obsluga bledow
            with open(modelsInfoURL, 'r') as f:
                try:
                    models = json.load(f)
                except json.JSONDecodeError:
                    print("Plik jest pusty lub niepoprawny.")
                    models = []
        except FileNotFoundError:
            print(modelsInfoURL)
            print("Brak pliku.")
            models = []

        return models

    def _model_exists(self, name):
        """
        Prywatna metoda, która sprawdza, czy dany model istnieje
        :param name:
        :return: boolean
        """
        return any(m['name'] == name for m in self._get_models_from_json())

    def _delete_from_json(self, model_name):
        """
        Prywatna metoda do usuwania modelu z jsona
        :param model_name:
        :return:
        """
        models = self._get_models_from_json() #pobranie listy
        filtered = [m for m in models if m['name'] != model_name] #filtrowanie po nazwie
        with open(modelsInfoURL, 'w') as f:
            json.dump(filtered, f, indent=2) #nadpisanie calego pliku

    def _add_to_the_json(self, modelName, modelPath, metrics, encoder_name: EncoderEnum):
        """
        Prywatna metoda do dodawania modelu do jsona
        :param modelName:
        :param modelPath:
        :param metrics:
        :return:
        """
        models = self._get_models_from_json()
        if self._model_exists(modelName): #jezeli model istnieje: usuniecie starego
            print("Model istnieje. Nadpisywanie...")
            self.delete_model(modelName)
            models = self._get_models_from_json()  # trzeba ponownie wczytac bo jest stara lista w tym momencie

        models.append({ #dodanie zaktualizowanego modelu
            "name": modelName,
            "metrics": metrics,
            "encoder_name": encoder_name.name
        })

        with open(modelsInfoURL, 'w') as f: #wgranie do pliku nowego
            json.dump(models, f, indent=2)
        print("Zapisano do pliku.")

    def add_model(self, model, model_name,metrics, encoder_name: EncoderEnum):
        """
        Metoda do dodania modelu
        :param encoder_name:
        :param model:
        :param model_name:
        :param metrics:
        :return:
        """
        #dodanie do jsona
        model_path = modelsFolder  #sciezka do nowego modelu
        self._add_to_the_json(model_name, model_path, metrics, encoder_name) #dodanie info do jsona

        #zapisz model
        os.makedirs(model_path, exist_ok=True) #wywolanie tego zapewnia ze folder istnieje
        joblib.dump(model, model_path / f"{model_name}.pkl") #dump modelu


    def delete_model(self, modelName):
        """
        Metoda do usuwania modelu
        :param modelName:
        :return:
        """
        if self._model_exists(modelName): #sprawdzenie czy model istnieje
            self._delete_from_json(modelName) #usuniecie z jsona
        else:
            print("Model o podanej nazwie nie istnieje.")
        pass