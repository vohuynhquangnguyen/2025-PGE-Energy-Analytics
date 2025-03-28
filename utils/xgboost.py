import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import os
import time

from utils.visualization import install_packages  # Reuse package installation from visualization

# Create directory for saving figures
output_dir = "./figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_and_prepare_data(train_path, test_path):
    """
    Load and prepare training and test data with feature engineering.
    """
    train_df = pd.read_excel(train_path, sheet_name='Data')
    test_df = pd.read_excel(test_path, sheet_name='Data')
    
    for df in [train_df, test_df]:
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    for df in [train_df, test_df]:
        df['avg_temp'] = df[[f'Site-{i} Temp' for i in range(1, 6)]].mean(axis=1)
        df['avg_ghi'] = df[[f'Site-{i} GHI' for i in range(1, 6)]].mean(axis=1)
    
    feature_cols = ['hour_sin', 'hour_cos', 'avg_temp', 'avg_ghi']
    
    for df in [train_df, test_df]:
        for col in feature_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    
    X_train_full = train_df[feature_cols]
    y_train_full = train_df['Load']
    X_test = test_df[feature_cols]
    y_test = test_df['Load']
    
    if X_test.shape[0] == 0:
        raise ValueError("Test set features are empty. Check your test data.")
    
    return X_train_full, y_train_full, X_test, y_test, feature_cols, train_df, test_df

def train_xgboost(X_train, y_train):
    """Train an XGBoost model with hyperparameter tuning using a pipeline."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    
    param_grid = {
        'xgb__n_estimators': [100, 200, 300],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    start_time = time.time()
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training the model took: {training_time:.4f} seconds")
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation R²:", grid.best_score_)
    
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test, subset_size=100, set_name='Validation'):
    """Evaluate the model and plot actual vs. predicted values."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    
    print(f"\n--- {set_name} Set Performance ---")
    print(f"{set_name} MAE: {mae:.2f}")
    print(f"{set_name} MSE: {mse:.2f}")
    print(f"{set_name} MAPE (%): {mape:.4f}")
    
    plt.figure(figsize=(15, 5))
    indices = np.arange(min(subset_size, len(y_test)))
    plt.plot(indices, y_test[:subset_size], label='Actual Load', marker='o', linestyle='-')
    plt.plot(indices, predictions[:subset_size], label='Predicted Load', marker='x', linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.title(f"{set_name} Set: Actual vs Predicted Load (First {subset_size} points)")
    plt.legend()
    output_filename = os.path.join(output_dir, f"{set_name.lower()}_actual_vs_predicted.pdf")
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    plt.show()

def generate_forecast(model, X_test, train_df, test_df, set_name='Test', y_test=None):
    """
    Generate forecast and plot it with historical data from Years 1 and 2.
    """
    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_excel('Results.xlsx', index=False, header=False)
    
    if y_test is not None and not (np.isnan(y_test).any()):
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        print(f"\n--- {set_name} Set Performance ---")
        print(f"{set_name} MAE: {mae:.2f}")
        print(f"{set_name} MSE: {mse:.2f}")
        print(f"{set_name} MAPE (%): {mape:.4f}")
    
    year1_length = len(train_df[train_df['Year'] == 1])
    year2_length = len(train_df[train_df['Year'] == 2])
    year3_length = len(test_df)
    subset_size = min(year1_length, year2_length, year3_length)
    
    year1_data = train_df[train_df['Year'] == 1].tail(subset_size)
    year2_data = train_df[train_df['Year'] == 2].tail(subset_size)
    forecast_data = predictions[:subset_size]
    
    total_train_points = len(train_df)
    year1_indices = np.arange(total_train_points - 2 * subset_size, total_train_points - subset_size)
    year2_indices = np.arange(total_train_points - subset_size, total_train_points)
    forecast_indices = np.arange(total_train_points, total_train_points + subset_size)
    
    plt.figure(figsize=(15, 5))
    plt.plot(year1_indices, year1_data['Load'], label='Year 1 Load', marker='o', linestyle='-', alpha=0.5)
    plt.plot(year2_indices, year2_data['Load'], label='Year 2 Load', marker='s', linestyle='-', alpha=0.5)
    plt.plot(forecast_indices, forecast_data, label='Forecasted Load (Year 3)', marker='x', linestyle='--')
    
    plt.xlabel("Time (hours relative to end of training data)")
    plt.ylabel("Load")
    plt.title(f"{set_name} Set: Historical (Years 1 & 2) and Forecasted Load (Last {subset_size} hours)")
    plt.legend()
    output_filename = os.path.join(output_dir, f"{set_name.lower()}_historical_vs_forecast.pdf")
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    plt.show()

def run_forecasting_pipeline():
    install_packages()
    train_path = './datasets/training.xlsx'
    test_path = './datasets/testing.xlsx'
    
    X_train_full, y_train_full, X_test, y_test, feature_cols, train_df, test_df = load_and_prepare_data(train_path, test_path)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, shuffle=False)
    
    model = train_xgboost(X_train, y_train)
    evaluate_model(model, X_val, y_val, set_name='Validation')
    generate_forecast(model, X_test, train_df, test_df, set_name='Test', y_test=y_test)
    
    model_filename = "best_XGBoost_optimized.pkl"
    joblib.dump(model, model_filename)
    print(f"Trained model saved as {model_filename}")

if __name__ == '__main__':
    run_forecasting_pipeline()
