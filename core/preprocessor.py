import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessor:

    def __init__(self, df: pd.DataFrame, target: str, task_type: str):
        self.df = df.copy()
        self.target = target
        self.task_type = task_type

        self.scaler = StandardScaler()
        self.label_encoders = {}

        self.X = None
        self.y = None

    def handle_missing(self):

        # numeric columns
        num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

        # categorical columns
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        self.df[cat_cols] = self.df[cat_cols].fillna(self.df[cat_cols].mode().iloc[0])
    def encode_categorical(self):
        for col in self.df.select_dtypes(include=["object"]):
            if col != self.target:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

    def split_features(self):
        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

    def scale_features(self):
        numeric_cols = self.X.select_dtypes(include=["int64", "float64"]).columns
        self.X[numeric_cols] = self.scaler.fit_transform(self.X[numeric_cols])

    def split_train_test(self, test_size=0.2):
        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=42
        )

    def process(self):
        self.handle_missing()
        self.encode_categorical()
        self.split_features()
        self.scale_features()

        return self.split_train_test()