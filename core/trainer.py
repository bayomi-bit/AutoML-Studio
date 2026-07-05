from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


class ModelTrainer:

    def __init__(self, task_type):
        self.task_type = task_type
        self.models = {}
        self.results = {}

    def build_models(self):

        if self.task_type == "classification":
            self.models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(),
                "SVM": SVC()
            }

        else:
            self.models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "SVR": SVR()
            }

    def train(self, X_train, X_test, y_train, y_test):

        self.build_models()

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if self.task_type == "classification":
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")

                self.results[name] = {
                    "Accuracy": acc,
                    "F1": f1
                }

            else:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)

                self.results[name] = {
                    "MSE": mse,
                    "R2": r2
                }

        return self.results

    def best_model(self):

        if self.task_type == "classification":
            return max(self.results, key=lambda x: self.results[x]["Accuracy"])
        else:
            return max(self.results, key=lambda x: self.results[x]["R2"])