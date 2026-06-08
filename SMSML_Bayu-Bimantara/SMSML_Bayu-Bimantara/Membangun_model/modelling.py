"""
modelling.py
Pelatihan model Iris Classification menggunakan MLflow autolog
Author: Bayu Bimantara
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# ── Konfigurasi MLflow ────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Iris-Classification-Bayu"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=['species'])
    y_train = train['species']
    X_test = test.drop(columns=['species'])
    y_test = test['species']

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def main():
    train_path = "iris_preprocessing/iris_train.csv"
    test_path = "iris_preprocessing/iris_test.csv"

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    with mlflow.start_run(run_name="RandomForest-autolog"):
        # Aktifkan autolog
        mlflow.sklearn.autolog()

        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred,
              target_names=['setosa', 'versicolor', 'virginica']))

        print("[INFO] Training selesai. Artefak tersimpan di MLflow Tracking UI.")


if __name__ == '__main__':
    main()
