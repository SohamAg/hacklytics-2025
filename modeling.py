from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import extract_features

# Function to load, preprocess, and feature engineer data
def prepare_data(file_path):
    # Load raw data
    data = load_data(file_path)

    # Preprocess the data
    data = preprocess_data(data)

    # Extract additional features
    data = extract_features(data)

    # Split data into features (X) and target (y)
    X, y = split_data(data)

    return X, y

# Function to build, tune, and evaluate models
def build_and_evaluate_model(X, y):
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Model
    rf_model = RandomForestClassifier(random_state=42)
    
    # Hyperparameter tuning with GridSearchCV
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    print("Best Random Forest Hyperparameters:", grid_search_rf.best_params_)

    # Gradient Boosting Model
    gb_model = GradientBoostingClassifier(random_state=42)
    
    param_grid_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    grid_search_gb = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)
    grid_search_gb.fit(X_train, y_train)
    best_gb_model = grid_search_gb.best_estimator_
    print("Best Gradient Boosting Hyperparameters:", grid_search_gb.best_params_)

    # Choose the best model
    best_model = best_rf_model if grid_search_rf.best_score_ > grid_search_gb.best_score_ else best_gb_model
    print(f"Using model: {best_model.__class__.__name__}")

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return best_model

# Example usage of the modeling function
if __name__ == "__main__":
    file_path = 'data/raw_data/penalty_data.csv'  # Replace with the actual path to your raw data
    X, y = prepare_data(file_path)
    best_model = build_and_evaluate_model(X, y)
