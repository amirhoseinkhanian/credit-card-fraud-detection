import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def load_best_params(params_file, model_name, data_type):
    """Load best parameters from JSON file."""
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            params = json.load(f)
        return params.get(data_type, {}).get(model_name, None)
    return None

def save_best_params(params_file, best_params, model_name, data_type):
    """Save best parameters to JSON file."""
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            all_params = json.load(f)
    else:
        all_params = {"balanced": {}, "imbalanced": {}}

    all_params[data_type][model_name] = best_params
    with open(params_file, 'w') as f:
        json.dump(all_params, f, indent=2)

def train_random_forest(X_train, y_train, params_file, data_type, random_state=42):
    """Train a Random Forest model with best parameters."""
    model_name = "Random Forest"
    best_params = load_best_params(params_file, model_name, data_type)

    if best_params is None:
        print(f"Running GridSearchCV for {model_name} ({data_type})...")
        rf = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best {model_name} Parameters: {best_params}")
        save_best_params(params_file, best_params, model_name, data_type)
        model = grid_search.best_estimator_
    else:
        print(f"Using pre-tuned {model_name} parameters: {best_params}")
        model = RandomForestClassifier(**best_params, random_state=random_state)
        model.fit(X_train, y_train)

    return model

def train_xgboost(X_train, y_train, params_file, data_type, random_state=42):
    """Train an XGBoost model with best parameters."""
    model_name = "XGBoost"
    best_params = load_best_params(params_file, model_name, data_type)

    if best_params is None:
        print(f"Running GridSearchCV for {model_name} ({data_type})...")
        xgb = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='f1', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best {model_name} Parameters: {best_params}")
        save_best_params(params_file, best_params, model_name, data_type)
        model = grid_search.best_estimator_
    else:
        print(f"Using pre-tuned {model_name} parameters: {best_params}")
        model = XGBClassifier(**best_params, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

    return model

def train_logistic_regression(X_train, y_train, params_file, data_type, random_state=42):
    """Train a Logistic Regression model with best parameters."""
    model_name = "Logistic Regression"
    best_params = load_best_params(params_file, model_name, data_type)

    if best_params is None:
        print(f"Running GridSearchCV for {model_name} ({data_type})...")
        lr = LogisticRegression(random_state=random_state, max_iter=1000)
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
        grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='f1', n_jobs=1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"Best {model_name} Parameters: {best_params}")
        save_best_params(params_file, best_params, model_name, data_type)
        model = grid_search.best_estimator_
    else:
        print(f"Using pre-tuned {model_name} parameters: {best_params}")
        model = LogisticRegression(**best_params, random_state=random_state, max_iter=1000)
        model.fit(X_train, y_train)

    return model