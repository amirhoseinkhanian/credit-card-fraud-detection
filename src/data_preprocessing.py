import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Load the credit card fraud dataset."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully. Shape:", data.shape)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

def explore_data(data):
    """Explore dataset structure and statistics."""
    print("\nColumn Information:")
    print(data.info())
    print("\nDescriptive Statistics:")
    print(data.describe())
    print("\nClass Distribution:")
    print(data['Class'].value_counts(normalize=True))
    print("\nMissing Values:")
    print(data.isnull().sum())

def preprocess_data_balanced(data, test_size=0.3, random_state=42):
    """Preprocess data: split, normalize based on training data, and balance classes with SMOTE."""
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize 'Time' and 'Amount' based on training data
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    # Balance training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print("\nClass Distribution after SMOTE (Balanced):")
    print(pd.Series(y_train_balanced).value_counts(normalize=True))

    return X_train_balanced, X_test, y_train_balanced, y_test

def preprocess_data_imbalanced(data, test_size=0.3, random_state=42):
    """Preprocess data: split and normalize based on training data, without balancing."""
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize 'Time' and 'Amount' based on training data
    scaler = StandardScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

    print("\nClass Distribution (Imbalanced):")
    print(pd.Series(y_train).value_counts(normalize=True))

    return X_train, X_test, y_train, y_test