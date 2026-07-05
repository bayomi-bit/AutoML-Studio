import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class Evaluator:

    def __init__(self, task_type):
        self.task_type = task_type

    def evaluate_classification(self, model, X_test, y_test, model_name):

        preds = model.predict(X_test)

        print(f"\n Report for {model_name}")
        print(classification_report(y_test, preds))

        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()

    def compare_models(self, results):

        models = list(results.keys())
        scores = [
            results[m]["Accuracy"] if "Accuracy" in results[m]
            else results[m]["R2"]
            for m in models
        ]

        plt.figure()
        plt.bar(models, scores)
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()