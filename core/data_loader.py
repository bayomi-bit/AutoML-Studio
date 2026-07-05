import pandas as pd


class DataLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    def shape(self):
        return self.df.shape

    def head(self, n=5):
        return self.df.head(n)

    def columns(self):
        return self.df.columns.tolist()

    def dtypes(self):
        return self.df.dtypes

    def missing_values(self):
        return self.df.isnull().sum()

    def duplicates(self):
        return self.df.duplicated().sum()

    def numerical_columns(self):
        return self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

    def categorical_columns(self):
        return self.df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def summary(self):
        return {
            "Rows": self.df.shape[0],
            "Columns": self.df.shape[1],
            "Duplicates": self.df.duplicated().sum(),
            "Numerical": len(self.numerical_columns()),
            "Categorical": len(self.categorical_columns())
        }