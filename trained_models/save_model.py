#Klasa pomocnicza do zapisywania modeli
#przyjmuje: nazwa modelu, model
#działanie - na podstawie przesłanych danych tworzy folder i dodaje tam pkl i dane o wytrenowanym modelu do jsona
import json
modelsInfoURL = './trained_models/models.json'

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

    @staticmethod
    def add_model(model, modelName):
        #zapisz model
        #zaktualizuj listę
        pass

    def delete_model(self, modelName):
        if self._model_exists(modelName):
            self._delete_from_json(modelName)
        else:
            print("Model o podanej nazwie nie istnieje.")
        #usun model
        #zaktualizuj listę
        #wazne: sprawdzic czy model istnieje
        pass