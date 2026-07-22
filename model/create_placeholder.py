import pickle
import numpy as np


class DummyModel:
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)
    
    def predict_proba(self, X):
        return np.array([[0.5, 0.5] for _ in range(X.shape[0])])


with open("model/smile_classifier.pkl", "wb") as f:
    pickle.dump(DummyModel(), f)

print("Placeholder model saved to model/smile_classifier.pkl")
print("Replace this with the actual trained model from Kaggle")