# SMSML - Bayu Bimantara
## Panduan Lengkap Menjalankan Submission

---

## Persiapan Environment

```bash
pip install -r Membangun_model/requirements.txt
```

---

## KRITERIA 1 - Eksperimen Dataset

### 1a. Buat repository GitHub
```
Eksperimen_SML_Bayu-Bimantara   (public)
├── .github/workflows/preprocessing.yml
├── iris_raw/
├── preprocessing/
│   ├── Eksperimen_Bayu-Bimantara.ipynb
│   └── automate_Bayu-Bimantara.py
└── iris_preprocessing/
```

### 1b. Jalankan notebook
Buka `preprocessing/Eksperimen_Bayu-Bimantara.ipynb` dan jalankan semua cell.

### 1c. Jalankan automate (Skilled)
```bash
cd preprocessing
python automate_Bayu-Bimantara.py
# Atau dengan argumen:
python automate_Bayu-Bimantara.py --output_dir ../iris_preprocessing --test_size 0.2
```

### 1d. GitHub Actions (Advance)
Push ke main branch → workflow otomatis berjalan dan meng-commit hasil preprocessing.

---

## KRITERIA 2 - Membangun Model

### 2a. Jalankan MLflow UI
```bash
mlflow ui --host 127.0.0.1 --port 5000
```
Buka browser: http://127.0.0.1:5000

### 2b. Basic - autolog
```bash
cd Membangun_model
python modelling.py
```

### 2c. Advance - manual logging + DagsHub + hyperparameter tuning
1. Daftar di https://dagshub.com dan buat repo baru
2. Edit `modelling_tuning.py` - ganti `DAGSHUB_OWNER` dan `DAGSHUB_REPO`
3. Login DagsHub:
   ```bash
   python -c "import dagshub; dagshub.auth.add_app_token('<TOKEN>')"
   ```
4. Jalankan:
   ```bash
   python modelling_tuning.py
   ```
5. Screenshot dashboard MLflow DagsHub dan simpan ke:
   - `screenshoot_dashboard.jpg`
   - `screenshoot_artifak.jpg`

---

## KRITERIA 3 - Workflow CI

### 3a. Buat repository GitHub baru
```
Workflow-CI   (public)
├── .github/workflows/ci.yml
└── MLProject/
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── Dockerfile
    └── iris_preprocessing/
```

### 3b. Tambahkan GitHub Secrets
Di repo → Settings → Secrets → Actions:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`
- `DAGSHUB_USER_TOKEN`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

### 3c. Jalankan MLProject secara lokal (tes dulu)
```bash
cd Workflow-CI/MLProject
mlflow run . --env-manager=local -P n_estimators=100
```

### 3d. Push ke GitHub → workflow CI otomatis berjalan

### 3e. Simpan tautan repo ke Workflow-CI.txt

---

## KRITERIA 4 - Monitoring & Logging

### 4a. Serve model
**Option A - MLflow serve:**
```bash
mlflow models serve -m runs:/<RUN_ID>/model -p 8080 --no-conda
```

**Option B - Docker (jika sudah push ke Docker Hub):**
```bash
docker pull <DOCKERHUB_USERNAME>/iris-model-bayu:latest
docker run -p 8080:8080 <DOCKERHUB_USERNAME>/iris-model-bayu:latest
```

Screenshot → simpan ke `1.bukti_serving`

### 4b. Jalankan Prometheus Exporter
```bash
cd "Monitoring dan Logging"
pip install prometheus-client scikit-learn
python 3.prometheus_exporter.py
```
Exporter berjalan di: http://localhost:8000/metrics

### 4c. Konfigurasi Prometheus
```bash
# Download Prometheus dari https://prometheus.io/download/
# Edit prometheus.yml sesuai file yang sudah disediakan
./prometheus --config.file=2.prometheus.yml
```
Prometheus UI: http://localhost:9090

Screenshot minimal 3 metrik → simpan ke `4.bukti monitoring Prometheus/`

### 4d. Konfigurasi Grafana
```bash
# Download Grafana dari https://grafana.com/grafana/download
# Atau gunakan Docker:
docker run -d -p 3000:3000 grafana/grafana
```
1. Buka http://localhost:3000 (admin/admin)
2. Tambah datasource Prometheus: http://localhost:9090
3. Buat dashboard dengan nama: **<username_dicoding>**
4. Tambahkan panels untuk 10+ metrik:
   - ml_requests_total
   - ml_request_latency_seconds
   - ml_prediction_class_total
   - ml_model_accuracy
   - ml_model_f1_score
   - ml_model_precision
   - ml_model_recall
   - ml_data_drift_score
   - ml_feature_mean
   - ml_feature_std
   - ml_errors_total
   - ml_uptime_seconds

Screenshot setiap panel → simpan ke `5.bukti monitoring Grafana/`

### 4e. Setup Alerting Grafana (3 alert untuk Advance)

**Alert 1: High Request Latency**
- Query: `histogram_quantile(0.95, ml_request_latency_seconds_bucket) > 0.5`
- Condition: IS ABOVE 0.5

**Alert 2: Data Drift Detected**
- Query: `ml_data_drift_score > 0.25`
- Condition: IS ABOVE 0.25

**Alert 3: Model Accuracy Drop**
- Query: `ml_model_accuracy < 0.90`
- Condition: IS BELOW 0.90

Screenshot rules dan notifikasi → simpan ke `6.bukti alerting Grafana/`

### 4f. Jalankan inference untuk generate data monitoring
```bash
python "Monitoring dan Logging/7.inference.py"
```

---

## Struktur File Submission Final

```
SMSML_Bayu-Bimantara.zip
├── Eksperimen_SML_Bayu-Bimantara.txt   ← tautan GitHub Kriteria 1
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── iris_preprocessing/
│   ├── screenshoot_dashboard.jpg
│   ├── screenshoot_artifak.jpg
│   ├── requirements.txt
│   └── DagsHub.txt
├── Workflow-CI.txt                      ← tautan GitHub Kriteria 3
└── Monitoring dan Logging/
    ├── 1.bukti_serving
    ├── 2.prometheus.yml
    ├── 3.prometheus_exporter.py
    ├── 4.bukti monitoring Prometheus/
    ├── 5.bukti monitoring Grafana/
    ├── 6.bukti alerting Grafana/
    └── 7.inference.py
```
