import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

warnings.filterwarnings("ignore", message=".*integer column.*")


df = pd.read_csv("data.csv")


df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%Y-%m-%d %H:%M:%S')

le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['Weather Condition'])


X = df[['Humidity (%)', 'Wind Speed (m/s)', 'Temperature_Normalized', 'Humidity_Normalized', 'Wind_Normalized', 'weather_encoded']]
y = df['Temp_Diff_From_Daily_Mean']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#mLflow run
with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2:", r2)

    input_example = X_test.iloc[:1]
    signature = infer_signature(X_test, y_pred)


    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model_type", "RandomForestRegressor")

    print("\n=== Predictions vs Actuals ===")
    for i, (pred, actual) in enumerate(zip(y_pred, y_test.values)):
        print(f"Sample {i+1}: Predicted = {pred:.2f}, Actual = {actual:.2f}")
