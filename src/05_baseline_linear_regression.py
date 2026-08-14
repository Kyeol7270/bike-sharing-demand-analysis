"""cnt(총 대여 건수) 예측 선형회귀 모델.
설명변수: instant, dteday, casual, registered 제외한 나머지 컬럼
  (casual/registered는 cnt=casual+registered 이므로 데이터 누수 방지 위해 제외)
데이터: data/day_processed.csv (hum=0 보간 반영본)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = "data/day_processed.csv"
MODEL_OUT = "models/linear_regression_baseline.joblib"
METRICS_OUT = "tables/linear_regression_cnt_metrics.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(DATA_PATH)

FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed"]
TARGET = "cnt"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"train size: {len(X_train)}, test size: {len(X_test)}")
print(f"R2   = {r2:.4f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAE  = {mae:.2f}")
print()
print("=== 회귀 계수 ===")
coef_df = pd.DataFrame({"feature": FEATURES, "coef": model.coef_})
coef_df = pd.concat(
    [pd.DataFrame({"feature": ["intercept"], "coef": [model.intercept_]}), coef_df],
    ignore_index=True,
)
print(coef_df.to_string(index=False))

joblib.dump(model, MODEL_OUT)

metrics_df = pd.DataFrame([{
    "model": "LinearRegression",
    "target": TARGET,
    "features": ",".join(FEATURES),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "R2": r2,
    "RMSE": rmse,
    "MAE": mae,
}])
metrics_df.to_csv(METRICS_OUT, index=False)

print()
print(f"saved model: {MODEL_OUT}")
print(f"saved metrics: {METRICS_OUT}")
