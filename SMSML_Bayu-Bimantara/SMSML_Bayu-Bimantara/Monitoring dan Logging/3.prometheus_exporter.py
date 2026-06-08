"""
prometheus_exporter.py
Custom Prometheus Exporter untuk monitoring ML model Iris
Author: Bayu Bimantara

Metrics yang dimonitor (10+ untuk Advance):
  1.  ml_requests_total              - total request prediksi
  2.  ml_request_latency_seconds     - latensi prediksi (histogram)
  3.  ml_prediction_class_total      - distribusi prediksi per kelas
  4.  ml_model_accuracy              - akurasi model saat ini
  5.  ml_model_f1_score              - F1-score model
  6.  ml_model_precision             - precision model
  7.  ml_model_recall                - recall model
  8.  ml_data_drift_score            - skor data drift (PSI)
  9.  ml_feature_mean                - rata-rata nilai fitur (per fitur)
  10. ml_feature_std                 - std nilai fitur (per fitur)
  11. ml_errors_total                - total error prediksi
  12. ml_uptime_seconds              - uptime exporter
"""

import time
import random
import threading
import numpy as np
import pandas as pd
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, REGISTRY
)
import requests
import json
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ── Konfigurasi ────────────────────────────────────────────────────────────────
EXPORTER_PORT = 8000
MODEL_SERVE_URL = "http://127.0.0.1:8080/invocations"
SCRAPE_INTERVAL = 5   # detik

# ── Prometheus Metrics Definitions ─────────────────────────────────────────────

# 1. Total request
ml_requests_total = Counter(
    'ml_requests_total',
    'Total number of prediction requests',
    ['model_name', 'status']
)

# 2. Request latency (histogram)
ml_request_latency = Histogram(
    'ml_request_latency_seconds',
    'Prediction request latency in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 3. Prediksi per kelas
ml_prediction_class = Counter(
    'ml_prediction_class_total',
    'Total predictions per class',
    ['model_name', 'class_name']
)

# 4. Model accuracy
ml_model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy on test set',
    ['model_name']
)

# 5. F1-Score
ml_model_f1 = Gauge(
    'ml_model_f1_score',
    'Current model F1-score (weighted)',
    ['model_name']
)

# 6. Precision
ml_model_precision = Gauge(
    'ml_model_precision',
    'Current model precision (weighted)',
    ['model_name']
)

# 7. Recall
ml_model_recall = Gauge(
    'ml_model_recall',
    'Current model recall (weighted)',
    ['model_name']
)

# 8. Data drift score (PSI - Population Stability Index)
ml_data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift score (PSI) per feature',
    ['feature_name']
)

# 9. Feature mean (nilai rata-rata fitur dari request terbaru)
ml_feature_mean = Gauge(
    'ml_feature_mean',
    'Rolling mean of feature values from recent requests',
    ['feature_name']
)

# 10. Feature std
ml_feature_std = Gauge(
    'ml_feature_std',
    'Rolling std of feature values from recent requests',
    ['feature_name']
)

# 11. Error total
ml_errors_total = Counter(
    'ml_errors_total',
    'Total number of prediction errors',
    ['model_name', 'error_type']
)

# 12. Uptime
ml_uptime_seconds = Gauge(
    'ml_uptime_seconds',
    'Exporter uptime in seconds'
)

# ── Data Setup ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'sepal length (cm)', 'sepal width (cm)',
    'petal length (cm)', 'petal width (cm)'
]
CLASS_NAMES = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
MODEL_NAME = "iris-rf-bayu"

# Load referensi data (training baseline untuk drift detection)
iris = load_iris()
scaler = StandardScaler()
X_ref = scaler.fit_transform(iris.data)

# Buffer request terbaru (sliding window 100 request)
recent_features = {f: [] for f in FEATURE_NAMES}
start_time = time.time()


def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Hitung Population Stability Index (PSI) untuk deteksi drift."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, buckets + 1)

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = (exp_counts + 0.0001) / len(expected)
    act_pct = (act_counts + 0.0001) / len(actual)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def simulate_prediction(features: list) -> dict:
    """
    Kirim request ke model serve, atau simulasi jika serve tidak aktif.
    """
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": [features]
        }
    }
    try:
        start = time.time()
        resp = requests.post(
            MODEL_SERVE_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        elapsed = time.time() - start
        resp.raise_for_status()
        result = resp.json()
        pred = result.get("predictions", [0])[0]
        return {"class": pred, "latency": elapsed, "status": "success"}
    except Exception:
        # Simulasi prediksi jika model belum serve
        elapsed = random.uniform(0.005, 0.08)
        pred = random.choices([0, 1, 2], weights=[0.33, 0.33, 0.34])[0]
        return {"class": pred, "latency": elapsed, "status": "simulated"}


def update_feature_stats(features: list):
    """Update rolling statistics fitur dari request terbaru."""
    for i, fname in enumerate(FEATURE_NAMES):
        recent_features[fname].append(features[i])
        # Sliding window: simpan 200 data terakhir
        if len(recent_features[fname]) > 200:
            recent_features[fname].pop(0)

        if len(recent_features[fname]) >= 5:
            vals = np.array(recent_features[fname])
            ml_feature_mean.labels(feature_name=fname).set(float(np.mean(vals)))
            ml_feature_std.labels(feature_name=fname).set(float(np.std(vals)))


def update_drift_scores():
    """Hitung PSI untuk setiap fitur."""
    for i, fname in enumerate(FEATURE_NAMES):
        if len(recent_features[fname]) >= 20:
            actual = np.array(recent_features[fname])
            expected = X_ref[:, i]
            psi = calculate_psi(expected, actual)
            ml_data_drift_score.labels(feature_name=fname).set(psi)


def update_model_metrics():
    """Simulasi update metrik model (dalam produksi, load dari MLflow)."""
    # Baca dari file jika ada, atau gunakan nilai default
    metrics_file = "model_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            m = json.load(f)
        acc = m.get("accuracy", 0.97)
        f1  = m.get("f1_score", 0.97)
        prec = m.get("precision", 0.97)
        rec  = m.get("recall", 0.97)
    else:
        # Simulasi dengan sedikit variasi
        acc  = 0.967 + random.uniform(-0.01, 0.01)
        f1   = 0.966 + random.uniform(-0.01, 0.01)
        prec = 0.968 + random.uniform(-0.01, 0.01)
        rec  = 0.967 + random.uniform(-0.01, 0.01)

    ml_model_accuracy.labels(model_name=MODEL_NAME).set(max(0, min(1, acc)))
    ml_model_f1.labels(model_name=MODEL_NAME).set(max(0, min(1, f1)))
    ml_model_precision.labels(model_name=MODEL_NAME).set(max(0, min(1, prec)))
    ml_model_recall.labels(model_name=MODEL_NAME).set(max(0, min(1, rec)))


def simulation_loop():
    """Loop utama: simulate request dan update semua metrics."""
    print(f"[INFO] Memulai simulation loop (interval: {SCRAPE_INTERVAL}s)")

    while True:
        # Generate sample request
        idx = random.randint(0, len(X_ref) - 1)
        features = X_ref[idx].tolist()

        # Simulasi prediksi
        result = simulate_prediction(features)

        # Update metrics
        status = result['status']
        ml_requests_total.labels(model_name=MODEL_NAME, status=status).inc()

        with ml_request_latency.labels(model_name=MODEL_NAME).time():
            time.sleep(result['latency'])

        class_name = CLASS_NAMES.get(result['class'], 'unknown')
        ml_prediction_class.labels(
            model_name=MODEL_NAME, class_name=class_name
        ).inc()

        # Update fitur stats
        update_feature_stats(features)

        # Update uptime
        ml_uptime_seconds.set(time.time() - start_time)

        # Setiap 10 iterasi, update model metrics dan drift
        if random.random() < 0.1:
            update_model_metrics()
            update_drift_scores()

        time.sleep(SCRAPE_INTERVAL)


def main():
    print("=" * 55)
    print("  Prometheus Exporter - Iris ML Monitoring")
    print(f"  Port: {EXPORTER_PORT}")
    print("  Author: Bayu Bimantara")
    print("=" * 55)

    # Set nilai awal model metrics
    update_model_metrics()

    # Inisialisasi gauge fitur
    for fname in FEATURE_NAMES:
        ml_feature_mean.labels(feature_name=fname).set(0)
        ml_feature_std.labels(feature_name=fname).set(0)
        ml_data_drift_score.labels(feature_name=fname).set(0)

    # Start Prometheus HTTP server
    start_http_server(EXPORTER_PORT)
    print(f"[INFO] Exporter berjalan di http://localhost:{EXPORTER_PORT}/metrics")
    print(f"[INFO] Tambahkan ke prometheus.yml:")
    print(f"       - targets: ['localhost:{EXPORTER_PORT}']")

    # Jalankan loop simulasi di thread terpisah
    t = threading.Thread(target=simulation_loop, daemon=True)
    t.start()

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Exporter dihentikan.")


if __name__ == '__main__':
    main()
