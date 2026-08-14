"""RandomForest 하이퍼파라미터 튜닝 (과적합 완화 목적, XGBoost 튜닝과 동일한 절차).
RandomizedSearchCV(5-fold CV)로 탐색 후, 튜닝 전/후 성능(train/test/CV) 비교.
데이터: data/day_processed.csv, 타겟: cnt (casual/registered 제외)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

DATA_PATH = "data/day_processed.csv"
MODEL_OUT = "models/RandomForest_tuned.joblib"
TABLE_OUT = "tables/random_forest_tuning_results.csv"

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


def cv_metrics(model):
    r2 = cross_val_score(model, X_full, y_full, cv=kf, scoring="r2")
    rmse = -cross_val_score(model, X_full, y_full, cv=kf, scoring="neg_root_mean_squared_error")
    mae = -cross_val_score(model, X_full, y_full, cv=kf, scoring="neg_mean_absolute_error")
    return r2.mean(), rmse.mean(), mae.mean()


# --- 튜닝 전 (기존 default 파라미터) 성능 ---
default_model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
default_model.fit(X_train, y_train)
cv_r2, cv_rmse, cv_mae = cv_metrics(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE))
before = {
    "stage": "before(default)",
    "train_R2": r2_score(y_train, default_model.predict(X_train)),
    "test_R2": r2_score(y_test, default_model.predict(X_test)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, default_model.predict(X_test))),
    "test_MAE": mean_absolute_error(y_test, default_model.predict(X_test)),
    "cv_R2_mean": cv_r2, "cv_RMSE_mean": cv_rmse, "cv_MAE_mean": cv_mae,
    "best_params": "n_estimators=300 (그 외 default)",
}

# --- 하이퍼파라미터 탐색 (과적합 완화 목적: 트리 깊이/분할 조건/변수 샘플링 제한) ---
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
}

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=60,
    scoring="r2",
    cv=kf,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
cv_r2, cv_rmse, cv_mae = cv_metrics(best_model)
after = {
    "stage": "after(tuned)",
    "train_R2": r2_score(y_train, best_model.predict(X_train)),
    "test_R2": r2_score(y_test, best_model.predict(X_test)),
    "test_RMSE": np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))),
    "test_MAE": mean_absolute_error(y_test, best_model.predict(X_test)),
    "cv_R2_mean": cv_r2, "cv_RMSE_mean": cv_rmse, "cv_MAE_mean": cv_mae,
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
