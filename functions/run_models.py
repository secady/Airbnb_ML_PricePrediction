from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
import xgboost
import catboost
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def run_models(X_train_scaled, X_test_scaled, y_train, y_test, path):
    models = [("RandomForest", RandomForestRegressor(random_state=42)),          
              ("AdaBoost", AdaBoostRegressor(random_state=42)),          
              ("GradientBoost", GradientBoostingRegressor(random_state=42)),          
              ("XGBoost", xgboost.XGBRegressor(random_state=42)),          
              ("CatBoost", catboost.CatBoostRegressor(random_state=42)),          
              ("KNN", KNeighborsRegressor())]
    
    metrics_df = pd.DataFrame(columns=["Model", "MSE", "RMSE", "MAE", "MAPE", "R2"])
    
    for name, model in models:
        print(f"Running {name} model...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = (name, mse, rmse, mae, mape, r2)
        metrics_df = metrics_df.append(pd.Series(metrics, index=metrics_df.columns), ignore_index=True)
    
    voting = VotingRegressor(estimators=models)
    voting.fit(X_train_scaled, y_train)
    y_pred = voting.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    voting_metrics = ("Voting", mse, rmse, mae, mape, r2)
    metrics_df = metrics_df.append(pd.Series(voting_metrics, index=metrics_df.columns), ignore_index=True)

    # Save metrics dataframe to CSV file
    metrics_df.to_csv(path, index=False)
    
    return metrics_df

