# Employee Attrition MLOps Pipeline

## Project Overview

This project implements a complete end-to-end MLOps pipeline for predicting employee attrition using the IBM HR Analytics Employee Attrition dataset.

The goal of the project is not only to train a machine learning model, but also to demonstrate production-style MLOps practices including:

- Version control with Git
- Data versioning with DVC
- Experiment tracking with MLflow
- Automated testing with pytest
- CI/CD pipelines with GitHub Actions
- Drift monitoring with Evidently AI

The project predicts whether an employee is likely to leave the company based on HR-related features such as age, salary, department, travel frequency, and job role.

---

# Dataset

Dataset:
IBM HR Analytics Employee Attrition Dataset

Source:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Task Type:
Binary classification

Target Variable:
`Attrition`

Classes:
- Yes
- No

The full raw dataset is tracked using DVC.

A smaller sample dataset is included for CI testing purposes.

---

# Repository Structure

```text
mlops-pipeline-project/
├── .github/workflows/
├── configs/
├── data/
│   ├── raw/
│   └── sample/
├── models/
├── reports/
├── src/
├── tests/
├── MONITORING.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Setup Instructions

## Clone Repository

```bash
git clone https://github.com/ketanguthula/mlops-pipeline-project.git
cd mlops-pipeline-project
```

## Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# DVC Setup

Initialize DVC:

```bash
dvc init
```

Pull tracked data:

```bash
dvc pull
```

Note:
GitHub Actions uses the sample dataset for CI testing because the full DVC remote is local.

---

# Model Training

Train using the full dataset:

```bash
python src/train.py
```

Train using CI configuration:

```bash
python src/train.py configs/ci_config.yaml
```

---

# MLflow Experiment Tracking

Launch MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow logs:
- model hyperparameters
- evaluation metrics
- experiment runs
- trained model artifacts

Experiment comparison script:

```bash
python src/compare_experiments.py
```

---

# Running Tests

Run the full pytest suite:

```bash
pytest tests/ -v
```

The test suite includes:
- preprocessing unit tests
- data validation tests
- model validation tests

---

# CI/CD Pipeline

GitHub Actions automatically runs:
1. pytest validation
2. model training

The training job depends on the testing job passing successfully.

Workflow file:

```text
.github/workflows/ci.yml
```

---

# Drift Monitoring

Run drift monitoring:

```bash
python src/monitor_drift.py
```

The script:
- compares reference vs simulated production data
- detects feature drift
- generates an HTML report
- exits with code 1 if drift exceeds the threshold

Generated report:

```text
reports/data_drift_report.html
```

Additional monitoring analysis is included in:

```text
MONITORING.md
```

---

# Technologies Used

- Python
- pandas
- scikit-learn
- MLflow
- DVC
- pytest
- Evidently AI
- GitHub Actions
- YAML

---

# Future Improvements

Potential future enhancements include:
- cloud-based DVC remote storage
- Docker containerization
- model deployment API
- automated retraining
- feature store integration
- advanced hyperparameter optimization
- real production monitoring dashboards