import pandas as pd
import os
from utils.config import DATA_DIR
from utils.preprocess import de_duplication, noise_remover, translate_to_en
from pipeline.feature_pipeline import FeaturePipeline

class DataLoader:
    """
    Loads and concatenates the AppGallery and Purchasing email datasets into a single DataFrame.
    Provides methods to extract labels and vectorize text using a TF-IDF pipeline.
    """

    def __init__(self, appgallery_filename="AppGallery.csv", purchasing_filename="Purchasing.csv"):
        self.appgallery_path = os.path.join(DATA_DIR, appgallery_filename)
        self.purchasing_path = os.path.join(DATA_DIR, purchasing_filename)
        self.pipeline = FeaturePipeline()

    def load_data(self):
        """
        Loads, cleans, and combines both email datasets into one DataFrame.
        Returns:
            pd.DataFrame: Combined and preprocessed email dataset.
        """
        print("Loading raw data...")
        appgallery_df = pd.read_csv(self.appgallery_path)
        purchasing_df = pd.read_csv(self.purchasing_path)

        appgallery_df = appgallery_df.drop(columns=['Unnamed: 11', 'Unnamed: 12'], errors='ignore')
        common_columns = purchasing_df.columns
        appgallery_df = appgallery_df[common_columns]

        combined_df = pd.concat([appgallery_df, purchasing_df], ignore_index=True)

        # Rename label columns for consistency with preprocess.py
        combined_df.rename(columns={
            "Type 1": "y1",
            "Type 2": "y2",
            "Type 3": "y3",
            "Type 4": "y4"
        }, inplace=True)

        essential_cols = ["Interaction content", "y2", "y3", "y4"]
        combined_df.dropna(subset=essential_cols, inplace=True)

        print("Deduplicating...")
        combined_df = de_duplication(combined_df)

        print("Removing noise...")
        combined_df = noise_remover(combined_df)

        print("Translating to English (this might take a while)...")
        combined_df['Interaction content'] = translate_to_en(combined_df['Interaction content'].tolist())

        # Rename back to original column names for compatibility with modeling code
        combined_df.rename(columns={
            "y1": "Type 1",
            "y2": "Type 2",
            "y3": "Type 3",
            "y4": "Type 4"
        }, inplace=True)

        return combined_df

    def load_preprocessed_data(self, texts, fit=True, pipeline=None):
        """
        Applies TF-IDF transformation using FeaturePipeline.
        """
        if pipeline is None:
            pipeline = self.pipeline
        return pipeline.fit_transform(texts) if fit else pipeline.transform(texts), pipeline
