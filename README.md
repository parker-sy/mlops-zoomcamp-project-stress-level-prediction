# MLOps Zoomcamp Project 2025 – Sleep Stress Prediction 🧘

## 📌 1. Problem Statement

In today's fast-paced world, sleep is often neglected despite its crucial role in physical and mental well-being. Chronic stress—often exacerbated by poor sleep quality—has become a major health concern. This project explores the relationship between various physiological indicators during sleep and stress levels the following day.

Using the [SaYoPillow dataset](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep/data), which simulates sensor data collected by a smart yoga pillow system, we train a machine learning model to predict next-day stress levels based on sleep-related physiological parameters. This solution supports the development of "smart sleeping" technologies and contributes to the broader vision of AI-assisted wellness.


## 📂 2. Dataset Description

The dataset `SayoPillow.csv` contains simulated sensor data generated from literature review. It includes the following features:

| Feature Name        | Description                                      | Example Value |
|---------------------|--------------------------------------------------|----------------|
| Snoring Range       | Intensity or frequency of snoring                | 3.0            |
| Respiration Rate    | Breaths per minute                               | 18             |
| Body Temperature    | Body temperature during sleep (°C)               | 36.7           |
| Limb Movement Rate  | Frequency of limb movements during sleep         | 20             |
| Blood Oxygen Levels | Oxygen saturation percentage                     | 97             |
| Eye Movement        | REM activity indicator                           | 1              |
| Hours of Sleep      | Total sleep duration in hours                    | 7.5            |
| Heart Rate          | Average heartbeats per minute                    | 65             |
| Stress Level        | Stress category (0–4)                            | 2              |

> 📎 Source: [Human Stress Detection in and through Sleep – Kaggle](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep/data)


## 🎯 3. Project Objectives

- Train and evaluate machine learning models using a reproducible pipeline
- Track experiments and register models
- Containerize and deploy the model locally or to the cloud
- Follow MLOps best practices (code quality, automation, CI/CD)


## ✅ 4. Self-Evaluation

| Criteria                         | Self-Evaluation                                                                 |
|----------------------------------|----------------------------------------------------------------------------------|
| Problem description              | ✅                                          |
| Cloud                            | ❌ Local development; deployment-ready for cloud.                                |
| Experiment tracking & registry   | ✅ MLflow used for experiment tracking and model registry                        |
| Workflow orchestration           | ✅ Training workflow managed with Prefect                                        |
| Model deployment                 | ✅ Dockerized deployment code; can be deployed locally or on the cloud           |
| Model monitoring                 | ❌ Not implemented                                                               |
| Reproducibility                  | ✅ Environment and workflow are fully reproducible (Pipenv and Docker ensure reproducibility)  |
| Best practices                   | ✅ Linter, code formatter, Makefile, pre-commit hooks, and CI/CD implemented  |


## 🔎 5. Experiment Tracking and Model Registry

```bash
# Start the MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db  --host localhost:5000

# Model training  (in a new terminal)
python Workflow/model_train_orchestrate.py

# Model registry
python Workflow/register_model.py
```


## 🚀 6. Model Deployment Instructions

### 🔧 6.1 Method 1: Quick Setup

```bash
# Clone the repository
git clone https://github.com/parker-sy/mlops-zoomcamp-project-stress-level-prediction.git
cd mlops-zoomcamp-project-stress-level-prediction

# Run Makefile to set up environment
make

# Build Docker image
docker build -f Deployment/Dockerfile -t sleep_stress_predictor .

# Run Docker container
docker run -v $(pwd):/app/ --network=host sleep_stress_predictor

# Refer to 6.2.5 for testing examples
```


### 🧪 6.2 Method 2: Step-by-Step Reproducibility

1. Create a virtual environment and install dependencies:
    ```bash
    pipenv install --python=3.12

    # Install dependencies - Option 1 (Recommended)
    pipenv install

    # Install dependencies - Option 2
    pipenv run pip install -r requirements.txt
    ```

2. Start the MLflow tracking server:
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db --host localhost:5000
    ```

3. Launch Prefect server:
    ```bash
    prefect server start
    ```

4. Run the application:
    - Option 1: Locally on http://127.0.0.1:5020/
        ```bash
        python app.py
        ```

    - Option 2: Using Docker
        ```bash
        docker build -f Deployment/Dockerfile -t sleep_stress_predictor .
        docker run -v $(pwd):/app/ --network=host sleep_stress_predictor
        ```

5. Testing examples
    - With Curl
    ```bash
    curl -X POST http://127.0.0.1:5020/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [93.8, 25.6, 91.8, 16.6, 89.84, 99.6, 1.84, 74.2]}'
     ```

     - With Python
    ```bash
    import requests

    url = 'http://127.0.0.1:5020/predict'
    data = {'features': [93.8, 25.6, 91.8, 16.6, 89.84, 99.6, 1.84, 74.2]}
    response = requests.post(url, json=data)
    print(response.json())
    ```
