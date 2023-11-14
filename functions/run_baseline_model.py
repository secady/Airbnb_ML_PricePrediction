from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def train_random_forest(X, y, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100, test_size=0.2, random_state=42, path1=str, path2=str):
    """
    Random Forest default hyperparameters:
    max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100
    path1 = working directory to save the feature importances
    path2 = working directory to save the metrics
    """

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a pipeline for scaling and modeling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=random_state, 
                                        max_depth=max_depth, 
                                        min_samples_leaf=min_samples_leaf, 
                                        min_samples_split=min_samples_split, 
                                        n_estimators=n_estimators))
    ])

    # Train the pipeline on the training set
    pipeline.fit(X_train, y_train)

    # Predict on test set
    y_pred_test = pipeline.predict(X_test)

    # Get the metrics for test data
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    r2_test = r2_score(y_test, y_pred_test)

    # Predict on train set
    y_pred_train = pipeline.predict(X_train)

    # Get the metrics for train data
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mape_train = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
    r2_train = r2_score(y_train, y_pred_train)

    # Get feature importances
    importances = pipeline.named_steps['model'].feature_importances_

    # Create a dataframe of feature importances
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    # Sort features by importance in descending order
    feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)
    feature_importances.to_csv(path1)

    # Add the metrics and feature importance to the metrics_df dataframe
    metrics_test = ["RandomForest_Test", mse_test, rmse_test, mae_test, mape_test, r2_test]
    metrics_train = ["RandomForest_Train", mse_train, rmse_train, mae_train, mape_train, r2_train]
    metrics_df = pd.DataFrame(columns=["Model", "MSE", "RMSE", "MAE", "MAPE", "R2"])
    metrics_df = metrics_df.append(pd.Series(metrics_test, index=metrics_df.columns), ignore_index=True)
    metrics_df = metrics_df.append(pd.Series(metrics_train, index=metrics_df.columns), ignore_index=True)
    metrics_df.to_csv(path2)

    return pipeline, metrics_df
