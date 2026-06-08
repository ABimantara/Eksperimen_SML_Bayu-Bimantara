"""
modelling.py - MLProject entry point
Iris Classification dengan MLflow manual logging
Author: Bayu Bimantara
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

TARGET_NAMES = ['setosa', 'versicolor', 'virginica']


def load_data():
    train_path = "iris_preprocessing/iris_train.csv"
    test_path  = "iris_preprocessing/iris_test.csv"

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    X_train = train.drop(columns=['species'])
    y_train = train['species']
    X_test  = test.drop(columns=['species'])
    y_test  = test['species']
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(feature_names)), imp[idx], color='steelblue')
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in idx], rotation=30, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main():
    # Baca hyperparameter dari environment (MLProject params)
    n_estimators   = int(os.environ.get("n_estimators", 100))
    max_depth_env  = os.environ.get("max_depth", "None")
    max_depth      = None if max_depth_env == "None" else int(max_depth_env)
    random_state   = int(os.environ.get("random_state", 42))

    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_train.columns)

    os.makedirs("artifacts", exist_ok=True)

    with mlflow.start_run():
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Metrik
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')
        f1        = f1_score(y_test, y_pred, average='weighted')
        logloss   = log_loss(y_test, y_prob)
        y_bin     = label_binarize(y_test, classes=[0, 1, 2])
        roc_auc   = roc_auc_score(y_bin, y_prob, average='weighted', multi_class='ovr')
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("roc_auc_weighted", roc_auc)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        # Artefak
        cm_path = "artifacts/confusion_matrix.png"
        fi_path = "artifacts/feature_importance.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_feature_importance(model, feature_names, fi_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        mlflow.set_tag("author", "Bayu Bimantara")
        mlflow.set_tag("dataset", "iris")

        print(f"[INFO] accuracy={accuracy:.4f}, f1={f1:.4f}, roc_auc={roc_auc:.4f}")
        print(f"[INFO] Run ID: {mlflow.active_run().info.run_id}")


if __name__ == '__main__':
    main()
