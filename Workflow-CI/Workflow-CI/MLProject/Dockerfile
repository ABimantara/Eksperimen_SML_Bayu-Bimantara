FROM python:3.12.7-slim

WORKDIR /app

COPY conda.yaml requirements.txt* ./
COPY modelling.py ./
COPY iris_preprocessing/ ./iris_preprocessing/

RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    dagshub==0.3.36

EXPOSE 5000 8080

CMD ["python", "modelling.py"]
