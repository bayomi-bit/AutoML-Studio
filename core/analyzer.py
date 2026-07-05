import pandas as pd


class DataAnalyzer:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.target = None
        self.task_type = None

    def guess_target(self):

        self.target = self.df.columns[-1]
        return self.target

    def detect_task_type(self):


        if self.target is None:
            self.guess_target()

        y = self.df[self.target]

        if y.dtype == "object" or y.nunique() < 20:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        return self.task_type

    def summary(self):
        return {
            "Target Column": self.target,
            "Task Type": self.task_type,
            "Rows": self.df.shape[0],
            "Columns": self.df.shape[1]
        }