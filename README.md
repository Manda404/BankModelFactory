---

# 🏦 BankModelFactory

End-to-end Machine Learning pipeline for predictive marketing in banking

---

## 📖 1. Project Overview

`BankModelFactory` is a **Data Science and MLOps project** designed to predict the likelihood that a banking client will subscribe to a **term deposit** based on marketing campaign data.

This project follows a **modular and production-grade architecture**, inspired by **Clean Architecture** and **MLOps best practices**, featuring:

* Clear separation of responsibilities (data, features, models, API)
* Full reproducibility using **Poetry**, **Hydra**, **MLflow**, and **Docker**
* Cloud-deployable **FastAPI** inference service
* Automated **CI/CD** with GitHub Actions and pre-commit hooks

---

## 💼 2. Business Context

Financial institutions frequently conduct **marketing campaigns** to promote banking products (like term deposits). However, these campaigns can be costly and inefficient if not properly targeted.

The dataset used in this project comes from the **[UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)**.
Each row represents a phone contact between a bank agent and a client. The goal is to predict whether the client will subscribe (`yes`) or not (`no`) to the offer.

### 🔍 Problem Statement

> How can we predict, *before contacting a client*, the probability that they will accept a term deposit offer — in order to reduce campaign costs and improve conversion rates?

---

## 🎯 3. Project Objective

The main goal is to build a **complete machine learning pipeline** capable of:

1. Performing exploratory analysis and data preprocessing
2. Training multiple classification algorithms (`Logistic Regression`, `Random Forest`, `CatBoost`, etc.)
3. Evaluating model performance using metrics like **ROC-AUC**, **PR-AUC**, and **F1-score**
4. Tracking experiments and results with **MLflow**
5. Deploying the final model as a **FastAPI REST API**
6. Running the entire system in the **cloud (Render, AWS ECS, or Railway)**

---

## 🧠 4. Solution Architecture

The solution is built around a **data-to-deployment pipeline**, following professional software engineering and MLOps principles.

---

### 🧩 Step 1 – Data Ingestion & Preparation

* Load and clean raw marketing data
* Handle missing values and unify data types
* Remove the `duration` variable (known data leakage)
* Split the dataset into train, validation, and test sets (stratified)

📁 Code: `src/bank_model_factory/data/`

---

### ⚙️ Step 2 – Feature Engineering

* Encode categorical variables (OneHot or CatBoost encoding)
* Impute missing values (median/mode)
* Scale numeric features (if needed)
* Implemented using Scikit-learn’s `Pipeline` and `ColumnTransformer`

📁 Code: `src/bank_model_factory/features/`

---

### 🧮 Step 3 – Modeling & Training

* Experiment with multiple models:

  * **Logistic Regression** (baseline)
  * **Random Forest**
  * **CatBoost**
  * **SVM**, **KNN**, **Naive Bayes**
  * *(optional)* **Artificial Neural Network (ANN)**
* Perform cross-validation (Stratified K-Fold)
* Log experiments and metrics using **MLflow**

📁 Code: `src/bank_model_factory/models/train.py`

---

### 📈 Step 4 – Evaluation & Interpretation

* Evaluate model performance with:

  * ROC-AUC, PR-AUC, F1, Precision, Recall
* Generate feature importance reports (SHAP values)
* Export visual reports to the `reports/` folder

📁 Code: `src/bank_model_factory/evaluation/`

---

### 🌐 Step 5 – API Deployment

* Expose a **REST API** using **FastAPI**
* `/predict` endpoint: takes JSON input, returns probability
* `/health` endpoint: for health checks
* Fully containerized with **Docker** for cloud deployment

📁 Code: `src/bank_model_factory/api/`

---

### ☁️ Step 6 – Industrialization & CI/CD

* Project packaged with **Poetry**
* Code formatting and linting enforced via **pre-commit hooks** (`black`, `ruff`, `isort`, `mypy`)
* Unit testing with **pytest** and **coverage**
* Continuous Integration using **GitHub Actions**
* Deployment-ready with **Dockerfile** and **docker-compose.yml**

---

## ⚙️ 5. Tech Stack

| Category            | Tools / Libraries                          |
| ------------------- | ------------------------------------------ |
| Data Processing     | Pandas, NumPy                              |
| Machine Learning    | Scikit-learn, CatBoost, PyTorch (optional) |
| Configuration       | Hydra, YAML                                |
| Experiment Tracking | MLflow                                     |
| API                 | FastAPI, Uvicorn                           |
| Packaging           | Poetry                                     |
| Code Quality        | Black, Ruff, Isort, Mypy                   |
| CI/CD               | GitHub Actions                             |
| Deployment          | Docker, Render / AWS ECS                   |

---

## 🧩 6. Project Structure

```
BankModelFactory/
├── configs/               # Hydra configuration files
├── data/                  # Raw and processed datasets
├── models/                # Trained models and artifacts
├── notebooks/             # Exploratory Data Analysis
├── reports/               # Reports and metrics
├── src/
│   ├── bank_model_factory/
│   │   ├── api/           # FastAPI inference service
│   │   ├── data/          # Data ingestion and processing
│   │   ├── features/      # Feature engineering
│   │   ├── models/        # Training and inference logic
│   │   ├── utils/         # Logging, config, I/O helpers
│   │   └── cli.py         # Typer command-line interface
│   └── scripts/           # Pipeline launcher and API runner
├── tests/                 # Unit tests
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 🎯 7. Final Goal

> Deliver a **robust, modular, and maintainable ML system** capable of:
>
> * Managing the full lifecycle of a predictive model (from data → model → API → cloud)
> * Serving as a **reference for modern Data Science and MLOps practices**
> * Being easily extended or deployed in real-world banking environments

---