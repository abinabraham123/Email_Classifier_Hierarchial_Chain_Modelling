from sklearn.feature_extraction.text import TfidfVectorizer

class FeaturePipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )

    def fit_transform(self, train_texts):
        """
        Fit the TF-IDF vectorizer on the training data and return the transformed data.
        """
        return self.vectorizer.fit_transform(train_texts)

    def transform(self, test_texts):
        """
        Transform the test data using the already fitted vectorizer.
        """
        return self.vectorizer.transform(test_texts)
