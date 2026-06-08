"""
modelling_tuning.py
Pelatihan model dengan hyperparameter tuning, manual logging MLflow, dan DagsHub
Author: Bayu Bimantara

Advance:
- Manual logging (bukan autolog)
- Hyperparameter tuning dengan GridSearchCV
- DagsHub sebagai remote MLflow tracking
- Minimal 2 artefak tambahan: confusion matrix plot + feature importance plot
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    log_loss
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
import warnings

warnings.filterwarnings('ignore')

# ── Konfigurasi DagsHub + MLflow ──────────────────────────────────────────────
DAGSHUB_OWNER = "bayubimantara"       # Ganti dengan username DagsHub Anda
DAGSHUB_REPO = "iris-msml"            # Ganti dengan nama repo DagsHub Anda
EXPERIMENT_NAME = "Iris-Tuning-Bayu"

dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment(EXPERIMENT_NAME)

TARGET_NAMES = ['setosa', 'versicolor', 'virginica']


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=['species'])
    y_train = train['species']
    X_test = test.drop(columns=['species'])
    y_test = test['species']

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, save_path: str):
    """Artefak tambahan 1: Confusion Matrix heatmap"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title('Confusion Matrix - Best Model', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Confusion matrix tersimpan: {save_path}")


def plot_feature_importance(model, feature_names, save_path: str):
    """Artefak tambahan 2: Feature Importance bar chart"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[indices], color='steelblue')
    plt.xticks(range(len(feature_names)),
               [feature_names[i] for i in indices], rotation=30, ha='right')
    plt.title('Feature Importance - Best Model', fontsize=14)
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Feature importance tersimpan: {save_path}")


def save_classification_report(y_test, y_pred, save_path: str):
    """Artefak tambahan 3: Classification report sebagai JSON"""
    report = classification_report(y_test, y_pred,
                                   target_names=TARGET_NAMES, output_dict=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Classification report tersimpan: {save_path}")


def hyperparameter_tuning(X_train, y_train):
    """GridSearchCV untuk menemukan hyperparameter terbaik"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }

    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n[INFO] Best params: {grid_search.best_params_}")
    print(f"[INFO] Best CV score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def main():
    train_path = "iris_preprocessing/iris_train.csv"
    test_path = "iris_preprocessing/iris_test.csv"
    artifact_dir = "artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(train_path, test_path)
    feature_names = list(X_train.columns)

    # Hyperparameter Tuning
    print("\n[INFO] Memulai hyperparameter tuning...")
    best_model, best_params, best_cv_score = hyperparameter_tuning(X_train, y_train)

    # Prediksi
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

    # ROC-AUC multiclass
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_bin, y_prob, average='weighted', multi_class='ovr')
    logloss = log_loss(y_test, y_prob)

    print(f"\n[INFO] Accuracy: {accuracy:.4f}")
    print(f"[INFO] Precision: {precision:.4f}")
    print(f"[INFO] Recall: {recall:.4f}")
    print(f"[INFO] F1-Score: {f1:.4f}")
    print(f"[INFO] ROC-AUC: {roc_auc:.4f}")
    print(f"[INFO] Log Loss: {logloss:.4f}")

    # Buat artefak lokal
    cm_path = os.path.join(artifact_dir, 'confusion_matrix.png')
    fi_path = os.path.join(artifact_dir, 'feature_importance.png')
    cr_path = os.path.join(artifact_dir, 'classification_report.json')
    model_path = os.path.join(artifact_dir, 'best_model.pkl')

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_feature_importance(best_model, feature_names, fi_path)
    save_classification_report(y_test, y_pred, cr_path)

    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # ── MLflow Manual Logging ──────────────────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest-Tuning-Manual"):

        # Log hyperparameter terbaik
        mlflow.log_params(best_params)

        # Log metrik (sama dengan autolog + tambahan)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted", recall)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("roc_auc_weighted", roc_auc)
        mlflow.log_metric("log_loss", logloss)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        mlflow.log_metric("best_cv_score", best_cv_score)
        mlflow.log_metric("training_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Log artefak tambahan 1: confusion matrix
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Log artefak tambahan 2: feature importance
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # Log artefak tambahan 3: classification report JSON
        mlflow.log_artifact(cr_path, artifact_path="reports")

        # Log artefak model pickle
        mlflow.log_artifact(model_path, artifact_path="model_files")

        # Log tags
        mlflow.set_tag("author", "Bayu Bimantara")
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("tuning_method", "GridSearchCV")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[INFO] MLflow Run ID: {run_id}")
        print("[INFO] Artefak tersimpan ke DagsHub MLflow Tracking.")


if __name__ == '__main__':
    main()
