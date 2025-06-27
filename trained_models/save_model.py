#Klasa pomocnicza do zapisywania modeli
#przyjmuje: nazwa modelu, model
#działanie - na podstawie przesłanych danych tworzy folder i dodaje tam pkl i dane o wytrenowanym modelu do jsona
import json, joblib, os, shutil
modelsFolder = "./trained_models"
modelsInfoURL = f'{modelsFolder}/models.json'

class ModelHandler:
    def _get_models_from_json(self):
        with open(modelsInfoURL) as f:
            models = json.load(f)
        return models

    def _model_exists(self, name):
        return any(m['name'] == name for m in self._get_models_from_json())

    def _delete_from_json(self, model_name):
        models = self._get_models_from_json()
        filtered = [m for m in models if m['name'] != model_name]
        with open(modelsInfoURL, 'w') as f:
            json.dump(filtered, f, indent=2)

    def _add_to_the_json(self, modelName, modelPath, metrics):
        try:
            with open(modelsInfoURL, 'r') as f:
                try:
                    models = json.load(f)
                except json.JSONDecodeError:
                    print("Plik jest pusty lub niepoprawny.")
                    models = []
        except FileNotFoundError:
            print("Brak pliku.")
            models = []
        if self._model_exists(modelName):
            print("Model istnieje. Nadpisywanie...")
            self.delete_model(modelName)
            models = self._get_models_from_json()  # trzeba ponownie wczyta

        models.append({
            "name": modelName,
            "path": modelPath,
            "metrics": metrics
        })

        with open(modelsInfoURL, 'w') as f:
            json.dump(models, f, indent=2)
        print("Zapisano do pliku.")

    def add_model(self, model, modelName, metrics):
        #dodanie do jsona
        model_path = f"{modelsFolder}/{modelName}"
        self._add_to_the_json(modelName, model_path, metrics)

        #zapisz model
        os.makedirs(model_path, exist_ok=True)
        # joblib.dump(model, './trained_models/logistic_regression_model.pkl')
        joblib.dump(model, f"{model_path}/{modelName}.pkl")

    @staticmethod
    def _delete_folder(modelName):
        path_to_delete = f"{modelsFolder}/{modelName}"

        if path_to_delete.startswith(modelsFolder):
            print(f"path: {path_to_delete}")
            shutil.rmtree(path_to_delete, ignore_errors=True)
        else:
            raise PermissionError("Ścieżka poza dozwolonym katalogiem")

    def delete_model(self, modelName):
        if self._model_exists(modelName):
            self._delete_from_json(modelName)
            self._delete_folder(modelName)
            #dodatkowa funkcja do sprawdzenia czy folder o takiej nazwie istnieje

        else:
            print("Model o podanej nazwie nie istnieje.")
        #usun model
        #zaktualizuj listę
        #wazne: sprawdzic czy model istnieje
        pass