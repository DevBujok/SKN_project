from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from data_handler.get_data import EncoderEnum, get_data
from trained_models.model_handler import ModelHandler

data, encoder = get_data(encoder=EncoderEnum.ONE_HOT_ENCODER, force_reload=True)

X = data.drop(['HeartDisease'], axis=1)
Y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics = classification_report(y_test, y_pred, output_dict=True)


rules = export_text(model, feature_names=list(X_train.columns))
print(rules)

modelHandler = ModelHandler()

modelHandler.add_model(model, "decision_tree", metrics, encoder)
