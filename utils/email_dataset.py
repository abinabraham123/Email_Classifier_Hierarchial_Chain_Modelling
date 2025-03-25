import pandas as pd
from sklearn.model_selection import train_test_split

class EmailDataset:
    def __init__(self, df, text_column='Interaction content', label_columns=None,
                 min_class_count=5, test_size=0.2, random_state=42):
        if label_columns is None:
            label_columns = ['Type 2', 'Type 3', 'Type 4']

        self.df = df.copy()
        self.text_column = text_column
        self.label_columns = label_columns
        self.min_class_count = min_class_count
        self.test_size = test_size
        self.random_state = random_state

        self.X_train, self.X_test, self.y_train, self.y_test = self._process()

    def _process(self):
        df = self.df.dropna(subset=[self.text_column] + self.label_columns)

        for label in self.label_columns:
            valid_labels = df[label].value_counts()
            valid_labels = valid_labels[valid_labels >= self.min_class_count].index
            df = df[df[label].isin(valid_labels)]

        X = df[self.text_column]
        y = df[self.label_columns]

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def get_train_test_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

