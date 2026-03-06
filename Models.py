import xgboost as xgb
import pandas as pd
import numpy as np
import sklearn as skl
import SmartContainer.Feature_Engineer as fe
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Model:
    def __init__(self, model = None):
        if model is None:
            self.model = xgb.XGBClassifier(n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss')
        else:
            self.model = model
    
    def train_model(self, X, y, test_size = 0.2, stratify = True):
        if stratify:
            stratify = y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
        skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
        accuracies = []
        for train_index, val_index in skf.split(X_train, y_train):
            X_train_split, X_val = X_train.iloc[train_index], X_train.iloc[val_index] 
            y_train_split, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

            self.model.fit(X_train_split, y_train_split)
            y_pred_val = self.model.predict(X_val)
            acc = accuracy_score(y_val, y_pred_val)
            accuracies.append(acc)

        print("Cross-validated accuracies:", accuracies)
        print("Mean accuracy:", np.mean(accuracies))

        y_pred = self.model.predict(X_test)

        print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Model Accuracy : ", accuracy_score(y_test, y_pred))
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        probs = self.model.predict_proba(X)
        scores = np.matmul(probs, np.unique(y_pred)) / (min(np.unique(y_pred)) + max(np.unique(y_pred)))
        return y_pred, probs, scores

    def new_model(self):
        return xgb.XGBClassifier()