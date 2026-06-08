"""
inference.py
Script untuk melakukan inference ke model yang di-serve via MLflow/FastAPI
Author: Bayu Bimantara
"""

import requests
import json
import numpy as np
import pandas as pd
import time

# ── Konfigurasi ────────────────────────────────────────────────────────────────
MODEL_SERVE_URL = "http://127.0.0.1:8080/invocations"  # mlflow models serve

# Contoh data Iris (sudah di-scale ≈ StandardScaler)
SAMPLE_DATA = [
    # sepal_length, sepal_width, petal_length, petal_width (sudah scaled)
    [-0.90, 1.02, -1.34, -1.31],   # setosa
    [ 1.03, -0.12,  0.82,  1.44],  # virginica
    [-0.17, -0.13,  0.26,  0.26],  # versicolor
    [-1.14,  0.79, -1.28, -1.31],  # setosa
    [ 0.55,  0.56,  1.27,  1.71],  # virginica
]
CLASS_NAMES = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
FEATURE_NAMES = [
    'sepal length (cm)', 'sepal width (cm)',
    'petal length (cm)', 'petal width (cm)'
]


def predict_single(data: list, url: str = MODEL_SERVE_URL) -> dict:
    """Kirim satu baris data ke endpoint inferensi."""
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": [data]
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Koneksi gagal. Pastikan model sudah di-serve."}
    except Exception as e:
        return {"error": str(e)}


def predict_batch(data_list: list, url: str = MODEL_SERVE_URL) -> dict:
    """Kirim batch data ke endpoint inferensi."""
    payload = {
        "dataframe_split": {
            "columns": FEATURE_NAMES,
            "data": data_list
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Koneksi gagal. Pastikan model sudah di-serve."}
    except Exception as e:
        return {"error": str(e)}


def health_check(url: str = "http://127.0.0.1:8080/ping") -> bool:
    """Cek apakah serving endpoint aktif."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"[OK] Model serving aktif di: {url}")
            return True
    except Exception:
        pass
    print(f"[WARN] Model serving tidak aktif di: {url}")
    return False


def run_load_test(n_requests: int = 50, url: str = MODEL_SERVE_URL):
    """Simulasi load test untuk keperluan monitoring Prometheus."""
    print(f"\n[INFO] Menjalankan load test: {n_requests} request...")
    success = 0
    latencies = []

    for i in range(n_requests):
        data = SAMPLE_DATA[i % len(SAMPLE_DATA)]
        start = time.time()
        result = predict_single(data, url)
        elapsed = (time.time() - start) * 1000  # ms

        if "error" not in result:
            success += 1
            latencies.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_requests}] sukses: {success}, "
                  f"avg latency: {np.mean(latencies):.1f}ms")

        time.sleep(0.1)  # throttle

    print(f"\n[HASIL] Total: {n_requests}, Sukses: {success}, Gagal: {n_requests - success}")
    if latencies:
        print(f"[HASIL] Latency - Avg: {np.mean(latencies):.1f}ms, "
              f"P95: {np.percentile(latencies, 95):.1f}ms, "
              f"Max: {max(latencies):.1f}ms")


def main():
    print("=" * 55)
    print("  INFERENCE SCRIPT - Iris Classification")
    print("  Author: Bayu Bimantara")
    print("=" * 55)

    # Health check
    is_up = health_check()
    if not is_up:
        print("\n[INFO] Jalankan model serving terlebih dahulu:")
        print("  mlflow models serve -m runs:/<RUN_ID>/model -p 8080 --no-conda")
        print("  atau: docker run -p 8080:8080 <image_name>")
        return

    # Single prediction
    print("\n--- Single Prediction ---")
    for i, sample in enumerate(SAMPLE_DATA):
        result = predict_single(sample)
        if "predictions" in result:
            pred_class = result["predictions"][0]
            print(f"  Sample {i+1}: {CLASS_NAMES.get(pred_class, pred_class)}")
        else:
            print(f"  Sample {i+1}: {result}")

    # Batch prediction
    print("\n--- Batch Prediction ---")
    batch_result = predict_batch(SAMPLE_DATA)
    if "predictions" in batch_result:
        for i, pred in enumerate(batch_result["predictions"]):
            print(f"  Sample {i+1}: {CLASS_NAMES.get(pred, pred)}")
    else:
        print(batch_result)

    # Load test (opsional, untuk generate monitoring data)
    print("\n--- Load Test (untuk Prometheus monitoring) ---")
    run_load_test(n_requests=30)


if __name__ == '__main__':
    main()
