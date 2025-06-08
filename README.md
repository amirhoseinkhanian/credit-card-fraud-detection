# Credit Card Fraud Detection

A modular machine learning pipeline to detect fraudulent credit card transactions using the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This project compares the performance of three models (Random Forest, XGBoost, Logistic Regression) on balanced and imbalanced data, evaluating prediction performance, training/inference time, and memory usage. Hyperparameters are optimized using GridSearchCV and stored for reuse to reduce computational overhead.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Project Structure

The project is organized as follows:

```plaintext
banking-anomaly-detection/
├── .gitignore                          # Excludes virtual env, dataset, and cache files
├── data/
│   └── creditcard.csv                  # Dataset file (excluded via .gitignore)
├── src/
│   ├── __init__.py                     # Makes src a Python package
│   ├── data_preprocessing.py           # Data loading and preprocessing (balanced and imbalanced)
│   ├── model_training.py               # Model training with tuned parameters
│   ├── evaluation.py                   # Model evaluation, visualization, and resource usage
│   └── main.py                         # Main script
├── notebooks/
│   └── exploration.ipynb               # Optional notebook for data exploration
├── results/
│   ├── model_comparison.txt            # Model comparison results
│   ├── roc_curve.png                   # ROC curve plot for balanced data
│   ├── precision_recall_curve.png      # Precision-Recall curve for balanced data
│   ├── roc_curve_imbalanced.png        # ROC curve for imbalanced data
│   ├── precision_recall_imbalanced.png # Precision-Recall curve for imbalanced data
│   ├── timing_comparison.png           # Training and inference time comparison
│   ├── memory_usage.png                # Memory usage comparison
│   ├── metrics_comparison.png          # Performance metrics comparison
│   └── best_params.json                # Best hyperparameters from GridSearchCV
├── README.md                           # Project documentation
├── requirements.txt                    # List of required libraries
└── LICENSE                             # License file

Dataset

Source: Kaggle Credit Card Fraud Detection
Description: Contains 284,807 transactions, with 492 frauds (0.172%).
Features: V1-V28 (PCA-transformed), Time, Amount, and Class (0 for non-fraud, 1 for fraud).
Note: The dataset (creditcard.csv) is excluded from the repository via .gitignore. You must download it manually and place it in the data/ directory.

Requirements

Python 3.8 or higher
Required libraries are listed in requirements.txt

Installation
Follow these steps to set up the project:

Clone the repository:
git clone https://github.com/Amirhoseinkhanian/credit-card-fraud-detection.git
cd banking-anomaly-detection


(Optional) Create and activate a virtual environment:
python -m venv bank
source bank/bin/activate  # On Windows: bank\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Download the dataset (creditcard.csv) from Kaggle and place it in the data/ directory.


Usage
Run the main script to execute the pipeline:
python src/main.py

The script will:

Load and preprocess the dataset
Train Random Forest, XGBoost, and Logistic Regression models on both balanced (SMOTE) and imbalanced data using pre-tuned hyperparameters from results/best_params.json
If best_params.json does not exist, perform GridSearchCV to find optimal hyperparameters and save them to best_params.json
Evaluate prediction performance (Precision, Recall, F1-Score, ROC-AUC, Average Precision, MCC)
Measure training/inference time and memory usage
Generate visualization plots for performance metrics, timing, and memory usage
Save results to results/model_comparison.txt, hyperparameters to results/best_params.json, and plots to results/

Methodology

Data Preprocessing:

Split data into training and testing sets
Normalize Time and Amount using training data statistics to avoid data leakage
For balanced models: Apply SMOTE to training data to handle class imbalance
For imbalanced models: Use raw training data without balancing


Model Training:

Train Random Forest, XGBoost, and Logistic Regression models
Use pre-tuned hyperparameters from results/best_params.json, obtained via GridSearchCV, to reduce computational cost
If hyperparameters are not available, perform GridSearchCV and save results to best_params.json
Train models separately on balanced (SMOTE) and imbalanced data


Evaluation:

Prediction Performance: Compare models using Precision, Recall, F1-Score, ROC-AUC, Average Precision, and Matthews Correlation Coefficient (MCC)
Operational Metrics: Measure training time, inference time, and memory usage (training and inference)
Visualizations: Generate ROC curves, Precision-Recall curves, timing comparison, memory usage comparison, and performance metrics comparison



Results

Metrics:
Detailed performance metrics are saved in results/model_comparison.txt, including:
Precision, Recall, F1-Score for the fraud class
ROC-AUC, Average Precision, and MCC
Training time, inference time, and memory usage


Optimal hyperparameters are saved in results/best_params.json, derived from GridSearchCV to minimize computational overhead in subsequent runs


Visualizations:
ROC curves: results/roc_curve.png (balanced) and results/roc_curve_imbalanced.png (imbalanced)
Precision-Recall curves: results/precision_recall_curve.png (balanced) and results/precision_recall_imbalanced.png (imbalanced)
Timing comparison: results/timing_comparison.png
Memory usage comparison: results/memory_usage.png
Performance metrics comparison: results/metrics_comparison.png


Insights:
Balanced models (with SMOTE) typically show higher recall and F1-score due to addressing class imbalance.
Imbalanced models may have higher precision but lower recall, reflecting the dataset's skewness.
Random Forest and XGBoost generally outperform Logistic Regression in prediction performance but may require more time and memory.
Using pre-tuned hyperparameters from best_params.json significantly reduces training time by avoiding repeated GridSearchCV.



License
This project is licensed under the MIT License. See the LICENSE file for details.```
