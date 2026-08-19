"""XGBoost 하이퍼파라미터 튜닝 (과적합 완화 목적).
RandomizedSearchCV(5-fold CV)로 탐색 후, 튜닝 전/후 성능(train/test/CV) 비교.
데이터: data/day_processed.csv, 타겟: cnt (casual/registered 제외)
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

DATA_PATH = "data/day_processed.csv"
MODEL_OUT = "models/xgboost_tuned_final.joblib"
TABLE_OUT = "tables/xgboost_tuning_results.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

df = pd.read_csv(DATA_PATH)
TARGET = "cnt"
BASE_FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
                  "weathersit", "temp", "atemp", "hum", "windspeed"]

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train, y_train = train_df[BASE_FEATURES], train_df[TARGET]
X_test, y_test = test_df[BASE_FEATURES], test_df[TARGET]
X_full, y_full = df[BASE_FEATURES], df[TARGET]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# --- 튜닝 전 (기존 default 파라미터) 성능 ---
default_model = XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0)
default_model.fit(X_train, y_train)
before = {
    "stage": "before(default)",
    "train_R2": r2_score(y_train, default_model.predict(X_train)),
    "test_R2": r2_score(y_test, default_model.predict(X_test)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, default_model.predict(X_test))),
    "test_MAE": mean_absolute_error(y_test, default_model.predict(X_test)),
    "cv_R2_mean": cross_val_score(XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
                                   X_full, y_full, cv=kf, scoring="r2").mean(),
    "best_params": "n_estimators=300 (그 외 default)",
}

# --- 하이퍼파라미터 탐색 (과적합 완화 목적: 트리 복잡도/학습률/샘플링 제한) ---
param_dist = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 1.5, 2, 3],
}

search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    param_distributions=param_dist,
    n_iter=60,
    scoring="r2",
    cv=kf,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
after = {
    "stage": "after(tuned)",
    "train_R2": r2_score(y_train, best_model.predict(X_train)),
    "test_R2": r2_score(y_test, best_model.predict(X_test)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))),
    "test_MAE": mean_absolute_error(y_test, best_model.predict(X_test)),
    "cv_R2_mean": cross_val_score(best_model, X_full, y_full, cv=kf, scoring="r2").mean(),
    "best_params": str(search.best_params_),
}

print("=== 튜닝 전 (default) ===")
for k, v in before.items():
    print(f"  {k}: {v}")
print()
print("=== 튜닝 후 (best) ===")
for k, v in after.items():
    print(f"  {k}: {v}")
print()
print(f"train-test R2 gap: before={before['train_R2']-before['test_R2']:.4f}  after={after['train_R2']-after['test_R2']:.4f}")

pd.DataFrame([before, after]).to_csv(TABLE_OUT, index=False)
joblib.dump(best_model, MODEL_OUT)
print()
print(f"saved comparison: {TABLE_OUT}")
print(f"saved tuned model: {MODEL_OUT}")
