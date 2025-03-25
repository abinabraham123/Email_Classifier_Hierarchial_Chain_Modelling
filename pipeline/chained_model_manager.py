from models.random_forest_model import RandomForestModel
from sklearn.metrics import classification_report
from utils.data_loader import DataLoader  # Still needed for TF-IDF pipeline

def run_chained_model_pipeline(dataset):
    # Get split data from the already-processed EmailDataset
    X_train_raw, X_test_raw, y_train, y_test = dataset.get_train_test_data()

    # Split labels for Type 2, 2+3, and 2+3+4
    y_type2_train = y_train[["Type 2"]]
    y_type2_test = y_test[["Type 2"]]

    y_type2_3_train = y_train[["Type 2", "Type 3"]].agg(" ".join, axis=1)
    y_type2_3_test = y_test[["Type 2", "Type 3"]].agg(" ".join, axis=1)

    y_type2_3_4_train = y_train[["Type 2", "Type 3", "Type 4"]].agg(" ".join, axis=1)
    y_type2_3_4_test = y_test[["Type 2", "Type 3", "Type 4"]].agg(" ".join, axis=1)

    # Feature transformation using DataLoader just for TF-IDF vectorizer
    tfidf_loader = DataLoader()  # we’re just using its pipeline utilities here
    X_train_vec, pipeline = tfidf_loader.load_preprocessed_data(X_train_raw, fit=True)
    X_test_vec, _ = tfidf_loader.load_preprocessed_data(X_test_raw, fit=False, pipeline=pipeline)

    # Train and evaluate each model on chained outputs
    print("\n--- Type 2 ---")
    model_2 = RandomForestModel()
    model_2.train(X_train_vec, y_type2_train.values.ravel())
    preds2 = model_2.predict(X_test_vec)
    model_2.print_results(y_type2_test, preds2)

    print("\n--- Type 2 + Type 3 ---")
    model_23 = RandomForestModel()
    model_23.train(X_train_vec, y_type2_3_train)
    preds23 = model_23.predict(X_test_vec)
    model_23.print_results(y_type2_3_test, preds23)

    print("\n--- Type 2 + Type 3 + Type 4 ---")
    model_234 = RandomForestModel()
    model_234.train(X_train_vec, y_type2_3_4_train)
    preds234 = model_234.predict(X_test_vec)
    model_234.print_results(y_type2_3_4_test, preds234)
