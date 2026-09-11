# Credit Card Fraud Detection System

> A machine learning system for identifying potentially fraudulent credit card transactions using imbalanced-learning techniques and fraud-focused evaluation metrics.

## Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions are extremely rare compared with legitimate transactions.

This project develops an end-to-end **Credit Card Fraud Detection System** that explores transaction patterns, handles severe class imbalance, compares multiple machine learning models, and evaluates their ability to detect fraudulent transactions while minimizing false alarms.

Rather than relying on accuracy alone, the project focuses on **Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix analysis** to provide a more meaningful assessment of fraud-detection performance.

## Problem Statement

In a real-world fraud detection system, missing a fraudulent transaction can be costly, but incorrectly blocking a legitimate transaction can also negatively affect customers.

The goal is therefore not simply:

> **"How accurately can we classify transactions?"**

but rather:

> **"How effectively can we identify fraudulent transactions while controlling false positives?"**

This project investigates that trade-off through model comparison and probability-threshold analysis.

---

## Project Objectives

* Analyze transaction-level patterns associated with fraud
* Explore the severe class imbalance in the dataset
* Perform data preprocessing and exploratory data analysis
* Build baseline classification models
* Apply techniques for handling imbalanced data
* Compare multiple machine learning algorithms
* Evaluate models using fraud-appropriate metrics
* Analyze precision-recall trade-offs
* Optimize the classification threshold
* Identify the most suitable model for the detection task
* Translate model performance into practical fraud-detection insights

---

## Dataset

The project uses the **Credit Card Fraud Detection dataset** containing transactions made by European cardholders.

### Dataset Characteristics

| Property                | Description            |
| ----------------------- | ---------------------- |
| Transactions            | 284,807                |
| Features                | 30                     |
| Fraudulent transactions | 492                    |
| Legitimate transactions | 284,315                |
| Fraud rate              | ~0.172%                |
| Target                  | `Class`                |
| `Class = 0`             | Legitimate transaction |
| `Class = 1`             | Fraudulent transaction |

The dataset contains anonymized PCA-transformed variables (`V1`–`V28`) along with `Time` and `Amount`.

> **Note:** The dataset is highly imbalanced. A model predicting almost every transaction as legitimate can achieve very high accuracy while failing to detect fraud.

---

## Machine Learning Workflow

```text
Raw Transaction Data
        │
        ▼
Data Inspection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Class Imbalance Analysis
        │
        ▼
Train / Test Split
        │
        ▼
Baseline Models
        │
        ▼
Imbalance Handling
        │
        ▼
Model Training & Comparison
        │
        ▼
Precision / Recall Analysis
        │
        ▼
Threshold Optimization
        │
        ▼
Final Model Selection
        │
        ▼
Fraud Prediction
```

---

## Exploratory Data Analysis

The analysis investigates:

* Distribution of legitimate vs fraudulent transactions
* Transaction amount distributions
* Transaction timing patterns
* Missing values and duplicate records
* Correlation between numerical features
* Fraud vs legitimate transaction characteristics
* Distribution of model-relevant features

### Key EDA Questions

Some of the questions explored include:

* How severe is the class imbalance?
* Do fraudulent transactions have different transaction amounts?
* Are fraudulent transactions concentrated during particular periods?
* Which features appear most informative?
* How does the distribution of fraud differ from legitimate activity?

---

## Handling Class Imbalance

Because fraudulent transactions represent only a tiny proportion of the dataset, class imbalance is a central part of the project.

The project investigates techniques such as:

* Class weighting
* SMOTE oversampling
* Stratified train-test splitting
* Threshold optimization

### Important Consideration

Resampling must be performed **only on the training data**.

Applying SMOTE before splitting the dataset can introduce information from the training process into the test set and lead to overly optimistic evaluation results.

---

## Models

The project compares multiple classification approaches.

### Baseline

**Logistic Regression**

Provides a simple and interpretable baseline for comparison.

### Tree-Based Models

**Random Forest**

Useful for capturing nonlinear relationships and interactions between features.

**XGBoost**

A gradient-boosting approach investigated for its ability to model complex patterns and its strong performance on structured/tabular data.

### Model Comparison

Each model is evaluated using the same held-out test data to determine how well it identifies fraudulent transactions.

---

## Evaluation Metrics

Accuracy is **not treated as the primary metric** because of the extreme class imbalance.

### Precision

Of all transactions predicted as fraud, how many were actually fraudulent?

$$
Precision = \frac{TP}{TP + FP}
$$

High precision means fewer legitimate customers are incorrectly flagged.

### Recall

Of all actual fraudulent transactions, how many did the model detect?

$$
Recall = \frac{TP}{TP + FN}
$$

High recall means fewer fraudulent transactions are missed.

### F1-Score

The harmonic mean of precision and recall.

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

### ROC-AUC

Measures the model's ability to distinguish between legitimate and fraudulent transactions across classification thresholds.

### PR-AUC

Precision-Recall AUC is particularly useful for this project because the positive class is extremely rare.

### Confusion Matrix

The confusion matrix provides:

* True Positives — correctly identified fraud
* True Negatives — correctly identified legitimate transactions
* False Positives — legitimate transactions incorrectly flagged
* False Negatives — fraudulent transactions missed by the model

---

## Threshold Optimization

A classification model typically converts predicted probabilities into classes using a threshold such as `0.50`.

However, fraud detection does not necessarily require a default 0.50 threshold.

For example:

```text
Lower Threshold
      │
      ├── More fraud detected
      ├── Higher recall
      └── More false positives

Higher Threshold
      │
      ├── Fewer false positives
      ├── Higher precision
      └── Potentially more missed fraud
```

This project evaluates different probability thresholds to understand the trade-off between **fraud detection and false alarms**.

The final threshold is selected based on the project's chosen evaluation objective rather than automatically assuming that `0.50` is optimal.

---

## Model Performance



| Model                              | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| ---------------------------------- | --------: | -----: | -------: | ------: | -----: |
| Logistic Regression                |         — |      — |        — |       — |      — |
| Logistic Regression + Class Weight |         — |      — |        — |       — |      — |
| Random Forest                      |         — |      — |        — |       — |      — |
| Random Forest + SMOTE              |         — |      — |        — |       — |      — |
| XGBoost                            |         — |      — |        — |       — |      — |
| **Final Model**                    |     **—** |  **—** |    **—** |   **—** |  **—** |

### Final Model

**Selected Model:** `YOUR MODEL`

**Decision Threshold:** `YOUR THRESHOLD`

**Precision:** `XX%`

**Recall:** `XX%`

**F1-Score:** `XX`

**ROC-AUC:** `XX`

**PR-AUC:** `XX`

The final model was selected based on its ability to balance fraud detection performance with the number of legitimate transactions incorrectly flagged.

---

## Key Insights

### 1. Accuracy can be misleading

Because fraud is extremely rare, a model can achieve very high accuracy without being useful for fraud detection.

This makes minority-class metrics much more important.

### 2. Fraud detection involves a trade-off

Increasing recall can help detect more fraudulent transactions, but may also increase false positives.

Increasing precision can reduce unnecessary alerts, but may cause more fraud cases to be missed.

### 3. Threshold selection matters

The default probability threshold is not necessarily appropriate for every business problem.

Adjusting the threshold allows the system to reflect different operational priorities.

### 4. Imbalance handling affects model behavior

Techniques such as class weighting and SMOTE can substantially change the model's ability to identify the minority class.

### 5. Model evaluation should reflect business costs

A fraud detection system should ultimately consider the relative cost of:

* Missing fraudulent transactions
* Investigating false alerts
* Blocking legitimate customers
* Operational investigation capacity

---

## Visualizations

The project includes visualizations such as:

* Class distribution
* Transaction amount distribution
* Fraud vs legitimate transaction comparison
* Correlation analysis
* Confusion matrices
* ROC curves
* Precision-Recall curves
* Model performance comparison
* Threshold optimization
* Feature importance

---

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
│
├── models/
│   └── final_model.pkl
│
├── figures/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── threshold_analysis.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The original dataset is not included in the repository because of its size. Download instructions are provided separately.

---

## Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Imbalanced-learn
* XGBoost

### Development

* Jupyter Notebook
* Git
* GitHub

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/credit-card-fraud-detection.git
```

Navigate into the project:

```bash
cd credit-card-fraud-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

1. Download the dataset from the original dataset source.
2. Place the CSV inside the appropriate `data/` directory.
3. Open the notebook:

```bash
jupyter notebook
```

4. Run the notebook from beginning to end.

The notebook covers:

```text
Data Loading
    ↓
Data Exploration
    ↓
Preprocessing
    ↓
Train/Test Split
    ↓
Imbalance Handling
    ↓
Model Training
    ↓
Model Comparison
    ↓
Threshold Optimization
    ↓
Final Evaluation
```

---

## Limitations

This project is a portfolio/research implementation and should not be considered a production-ready banking fraud system.

Important limitations include:

* Dataset features are largely anonymized.
* Real-world fraud patterns can change over time.
* Model performance may degrade under data drift.
* The dataset does not represent every type of financial fraud.
* Production systems would require real-time transaction processing.
* Business-specific fraud costs would need to be incorporated into threshold selection.
* Additional monitoring and model explainability would be required for production deployment.

---

## Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Cost-sensitive learning
* Advanced threshold optimization
* SHAP-based model explainability
* Anomaly detection
* Real-time prediction API
* Streamlit dashboard
* Transaction-level risk scoring
* Model monitoring
* Data-drift detection
* Automated retraining pipeline
* Docker deployment
* Cloud deployment

---

## Portfolio Value

This project demonstrates practical experience with:

* **End-to-end machine learning workflows**
* **Binary classification**
* **Highly imbalanced datasets**
* **Data preprocessing**
* **Exploratory data analysis**
* **Model comparison**
* **Precision-recall trade-offs**
* **Threshold optimization**
* **Model evaluation**
* **Business-oriented interpretation of ML results**

The project emphasizes not only building a predictive model, but also understanding **whether the model's predictions would be useful in a real fraud-detection scenario**.

---

## Disclaimer

This project is intended for educational and portfolio purposes. It does not represent a production financial fraud detection system and should not be used to make real financial decisions.

## Author

**Rojina Dhakal**

Aspiring Data Analyst / Data Scientist

* Python
* SQL
* Excel
* Data Analysis
* Machine Learning
* Data Visualization
