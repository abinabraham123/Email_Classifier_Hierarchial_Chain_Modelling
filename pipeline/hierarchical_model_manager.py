from models.random_forest_model import RandomForestModel
from utils.data_loader import DataLoader  # for TF-IDF only
from sklearn.metrics import classification_report
import pandas as pd

def run_hierarchical_model_pipeline(dataset):
    # Get pre-split data from EmailDataset
    X_train_raw, X_test_raw, y_train, y_test = dataset.get_train_test_data()

    # Split each label column
    y2_train = y_train["Type 2"]
    y2_test = y_test["Type 2"]

    y3_test = y_test["Type 3"]
    y4_test = y_test["Type 4"]

    # Feature transformation using TF-IDF
    tfidf_loader = DataLoader()  # for vectorization only
    X_train_vec, pipeline = tfidf_loader.load_preprocessed_data(X_train_raw, fit=True)
    X_test_vec, _ = tfidf_loader.load_preprocessed_data(X_test_raw, fit=False, pipeline=pipeline)

    # Stage 1: Train and evaluate on Type 2
    print("\n--- Hierarchical Stage 1: Type 2 ---")
    model_t2 = RandomForestModel()
    model_t2.train(X_train_vec, y2_train)
    preds_t2 = model_t2.predict(X_test_vec)
    model_t2.print_results(y2_test, preds_t2)

    # Align prediction outputs
    df_test = pd.DataFrame({
        "X": list(X_test_raw),
        "Pred_Type2": preds_t2,
        "True_Type3": list(y3_test),
        "True_Type4": list(y4_test)
    })

    # Stage 2: Type 3 classification conditioned on Pred_Type2
    print("\n--- Hierarchical Stage 2: Type 3 (per Type 2 class) ---")
    for t2_class in df_test["Pred_Type2"].unique():
        subset_t2 = df_test[df_test["Pred_Type2"] == t2_class]
        if len(subset_t2) < 2:
            continue

        X_vec_t3 = pipeline.transform(subset_t2["X"])
        y_t3 = subset_t2["True_Type3"]

        model_t3 = RandomForestModel()
        model_t3.train(X_vec_t3, y_t3)
        pred_t3 = model_t3.predict(X_vec_t3)
        model_t3.print_results(y_t3, pred_t3)

        # Stage 3: Type 4 classification conditioned on Pred_Type3
        print(f"\n--- Hierarchical Stage 3: Type 4 (under Type2={t2_class}) ---")
        subset_t2 = subset_t2.copy()  # to avoid SettingWithCopyWarning
        subset_t2["Pred_Type3"] = pred_t3

        for t3_class in subset_t2["Pred_Type3"].unique():
            subset_t3 = subset_t2[subset_t2["Pred_Type3"] == t3_class]
            if len(subset_t3) < 2:
                continue

            X_vec_t4 = pipeline.transform(subset_t3["X"])
            y_t4 = subset_t3["True_Type4"]

            model_t4 = RandomForestModel()
            model_t4.train(X_vec_t4, y_t4)
            pred_t4 = model_t4.predict(X_vec_t4)
            model_t4.print_results(y_t4, pred_t4)
