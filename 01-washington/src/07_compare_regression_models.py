"""cnt 예측 모델 비교: 선형회귀(순서형 인코딩, baseline) vs 선형회귀(원-핫 인코딩)
   vs RandomForest vs XGBoost.
설명변수: instant, dteday, casual, registered 제외 (casual/registered는 데이터 누수 방지).
데이터: data/day_processed.csv (hum=0 보간 반영본)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = "data/day_processed.csv"
MODEL_DIR = "models"
TABLE_OUT = "tables/model_comparison.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

df = pd.read_csv(DATA_PATH)

TARGET = "cnt"
BASE_FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
                  "weathersit", "temp", "atemp", "hum", "windspeed"]
CAT_FOR_ONEHOT = ["season", "mnth", "weekday", "weathersit"]

# 동일한 train/test 행을 모든 모델에 공통 적용
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
y_train, y_test = train_df[TARGET], test_df[TARGET]

# 원-핫 인코딩 (전체 df 기준으로 인코딩 후 동일 인덱스로 분할 -> train/test 컬럼 일치 보장)
df_onehot = pd.get_dummies(df, columns=CAT_FOR_ONEHOT, drop_first=True)
onehot_features = [c for c in df_onehot.columns
                    if c not in ["instant", "dteday", "casual", "registered", TARGET]]
X_train_oh = df_onehot.loc[train_df.index, onehot_features]
X_test_oh = df_onehot.loc[test_df.index, onehot_features]

results = []
fitted_models = {}


def evaluate(name, model, X_train, X_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    results.append({"model": name, "n_features": X_train.shape[1],
                     "R2": r2, "RMSE": rmse, "MAE": mae})
    fitted_models[name] = model
    print(f"[{name}] features={X_train.shape[1]}  R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")


# 1. 선형회귀 baseline (순서형 인코딩, 11개 변수)
evaluate("LinearRegression_ordinal", LinearRegression(),
          train_df[BASE_FEATURES], test_df[BASE_FEATURES])

# 2. 선형회귀 (원-핫 인코딩)
evaluate("LinearRegression_onehot", LinearRegression(), X_train_oh, X_test_oh)

# 3. Random Forest (순서형 인코딩 그대로 사용 - 트리 모델은 인코딩 불필요)
evaluate("RandomForest", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
          train_df[BASE_FEATURES], test_df[BASE_FEATURES])

# 4. XGBoost
evaluate("XGBoost", XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
          train_df[BASE_FEATURES], test_df[BASE_FEATURES])

results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print()
print(results_df.to_string(index=False))

results_df.to_csv(TABLE_OUT, index=False)
for name, model in fitted_models.items():
    joblib.dump(model, f"{MODEL_DIR}/{name}.joblib")

print()
print(f"saved comparison table: {TABLE_OUT}")
print(f"saved models to: {MODEL_DIR}/")
