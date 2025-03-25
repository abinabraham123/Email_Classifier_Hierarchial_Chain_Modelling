import pandas as pd
import os
from utils.config import DATA_DIR
from pipeline.feature_pipeline import FeaturePipeline

class DataLoader:
    """
    Loads and concatenates the AppGallery and Purchasing email datasets into a single DataFrame.
    Provides methods to extract labels and vectorize text using a TF-IDF pipeline.
    """

    def __init__(self, appgallery_filename="AppGallery.csv", purchasing_filename="Purchasing.csv"):
        self.appgallery_path = os.path.join(DATA_DIR, appgallery_filename)
        self.purchasing_path = os.path.join(DATA_DIR, purchasing_filename)

    def load_data(self):
        """
        Loads, cleans, and combines both email datasets into one DataFrame.
        Returns:
            pd.DataFrame: Combined and preprocessed email dataset.
        """
        # Load the datasets
        appgallery_df = pd.read_csv(self.appgallery_path)
        purchasing_df = pd.read_csv(self.purchasing_path)

        # Drop unused columns from AppGallery if they exist
        appgallery_df = appgallery_df.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore')

        # Align columns between datasets
        common_columns = purchasing_df.columns
        appgallery_df = appgallery_df[common_columns]

        # Concatenate the two datasets
        combined_df = pd.concat([appgallery_df, purchasing_df], ignore_index=True)

        # Drop rows missing essential classification inputs
        essential_cols = ["Interaction content", "Type 2", "Type 3", "Type 4"]
        combined_df.dropna(subset=essential_cols, inplace=True)

        return combined_df

    def get_text_and_labels(self, df):
        """
        Splits the DataFrame into text (X) and label (y) parts.
        Args:
            df (pd.DataFrame): Preprocessed DataFrame.
        Returns:
            X_raw (pd.Series): Raw email content.
            y (pd.DataFrame): Multi-label classification targets.
        """
        X_raw = df["Interaction content"]
        y = df[["Type 2", "Type 3", "Type 4"]]
        return X_raw, y

    def load_preprocessed_data(self, X_raw, fit=True, pipeline=None):
        """
        Vectorizes raw text using TF-IDF.
        Args:
        X_raw (pd.Series): Raw email text.
        fit (bool): Whether to fit the vectorizer or just transform.
        pipeline (FeaturePipeline): Optional pre-fitted pipeline.
        Returns:
        X_vec: TF-IDF vectorized output.
        pipeline: The fitted or reused FeaturePipeline instance.
        """
        if pipeline is None:
            pipeline = FeaturePipeline()

        if fit:
            X_vec = pipeline.fit_transform(X_raw)
        else:
            X_vec = pipeline.transform(X_raw)

        return X_vec, pipeline


# Example usage for standalone testing
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    X_raw, y = loader.get_text_and_labels(df)
    X_vec, pipeline = loader.load_preprocessed_data(X_raw, fit=True)

    print("Combined dataset shape:", df.shape)
    print("Sample labels:", y.head())
    print("TF-IDF vectorized shape:", X_vec.shape)
