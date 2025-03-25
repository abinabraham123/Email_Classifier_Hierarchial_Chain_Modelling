from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from models.base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def print_results(self, y_true, y_pred):
        print(classification_report(y_true, y_pred, zero_division=0))