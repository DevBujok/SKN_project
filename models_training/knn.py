from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from data_handler.get_data import EncoderEnum, get_data
from trained_models.model_handler import ModelHandler
from sklearn.neighbors import KNeighborsClassifier

chosen_encoder = EncoderEnum.ONE_HOT_ENCODER

data, encoder = get_data(encoder=chosen_encoder, force_reload=True)

X = data.drop(['HeartDisease'], axis=1)
Y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = classification_report(y_test, y_pred, output_dict=True)

modelHandler = ModelHandler()

modelHandler.add_model(model, "knn_5", metrics, chosen_encoder)
