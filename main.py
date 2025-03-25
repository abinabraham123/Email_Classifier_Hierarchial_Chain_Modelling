import os
from utils.data_loader import DataLoader
from utils.email_dataset import EmailDataset
from pipeline.chained_model_manager import run_chained_model_pipeline
from pipeline.hierarchical_model_manager import run_hierarchical_model_pipeline




def main():
    # Resolve the absolute path to the CSV file
    print("Loading data...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    loader = DataLoader()  # default loads both AppGallery & Purchasing
    df = loader.load_data()
    dataset = EmailDataset(df, label_columns=['Type 2', 'Type 3', 'Type 4'])


    # --- Chained Multi-Output Modeling ---
    print("\n--- Running Chained Multi-Output Modeling ---")
    run_chained_model_pipeline(dataset)

    # --- Hierarchical Modeling Execution ---
    print("\n--- Running Hierarchical Modeling ---")
    # Then inside main():
    run_hierarchical_model_pipeline(dataset)

    


if __name__ == '__main__':
    main()

