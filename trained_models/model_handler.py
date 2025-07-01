#Klasa pomocnicza do zapisywania modeli
#przyjmuje: nazwa modelu, model
#działanie - na podstawie przesłanych danych tworzy folder i dodaje tam pkl i dane o wytrenowanym modelu do jsona
import json, joblib, os, shutil
from pathlib import Path

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

    def _add_to_the_json(self, modelName, modelPath, metrics):
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
            "metrics": metrics
        })

        with open(modelsInfoURL, 'w') as f: #wgranie do pliku nowego
            json.dump(models, f, indent=2)
        print("Zapisano do pliku.")

    def add_model(self, model, modelName,metrics, labelEncoder, prescaler = None):
        """
        Metoda do dodania modelu
        :param model:
        :param modelName:
        :param metrics:
        :return:
        """
        #dodanie do jsona
        model_path = modelsFolder / f"{modelName}" #sciezka do nowego modelu
        self._add_to_the_json(modelName, model_path, metrics) #dodanie info do jsona

        #zapisz model
        os.makedirs(model_path, exist_ok=True) #wywolanie tego zapewnia ze folder istnieje
        joblib.dump(model, model_path / f"{modelName}.pkl") #dump modelu
        joblib.dump(labelEncoder, model_path / f"{modelName}_label_encoder.pkl")
        if prescaler is not None:
            joblib.dump(prescaler, model_path/ f"{modelName}_prescaler.pkl")

    @staticmethod
    def _delete_folder(modelName):
        """
        Prywatna statyczna metoda do usuniecia istniejacego folderu
        :param modelName:
        :return:
        """
        path_to_delete = modelsFolder / f"{modelName}" #sciezka do usuniecia

        if path_to_delete.resolve().is_relative_to(modelsFolder.resolve()): #dodatkowe sprawdzenie czy sciezka znajduje sie w katalogu domowym,
            # bo ona moze usuwac wszystko globalnie
            shutil.rmtree(path_to_delete, ignore_errors=True) #usuniecie
        else:
            raise PermissionError("Ścieżka poza dozwolonym katalogiem")

    def delete_model(self, modelName):
        """
        Metoda do usuwania modelu
        :param modelName:
        :return:
        """
        if self._model_exists(modelName): #sprawdzenie czy model istnieje
            self._delete_from_json(modelName) #usuniecie z jsona
            self._delete_folder(modelName) #usuniecie folderu
        else:
            print("Model o podanej nazwie nie istnieje.")
        pass