from core.analyzer import DataAnalyzer
from core.preprocessor import Preprocessor
from core.trainer import ModelTrainer


class AutoMLPipeline:

    def __init__(self):
        self.target = None
        self.task = None

    def run(self, df):

        # 1. analyze
        analyzer = DataAnalyzer(df)
        self.target = analyzer.guess_target()
        self.task = analyzer.detect_task_type()

        # 2. preprocess
        prep = Preprocessor(df, self.target, self.task)
        X_train, X_test, y_train, y_test = prep.process()

        # 3. train
        trainer = ModelTrainer(self.task)
        results = trainer.train(X_train, X_test, y_train, y_test)

        best_model = trainer.best_model()
        model_obj = trainer.models[best_model]

        return {
            "results": results,
            "best_model": best_model,
            "model": model_obj,
            "data": (X_test, y_test)
        }