"""RandomForest 하이퍼파라미터 튜닝. ANALYSIS_PLAN.md 3-5절.
RandomizedSearchCV(5-fold CV, n_iter=60), train(2015) 내부에서만 진행 (2016 미사용).
n_jobs=1 고정 (사유는 08_tune_xgboost.py와 동일 — 경로 한글로 인한 joblib 멀티프로세싱 이슈).
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

TRAIN_PATH = "data/day_2015.csv"
MODEL_OUT = "models/random_forest_tuned.joblib"
RANDOM_STATE = 42

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
TREE_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                  "weather_code", "t1", "t2", "hum", "wind_speed"]

X = train[TREE_FEATURES]
y = train[TARGET]

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 3, 5, 8, 12, 16],
    "min_samples_split": [2, 4, 6, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_model = RandomForestRegressor(random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=60, cv=kf,
    scoring="r2", random_state=RANDOM_STATE, n_jobs=1, verbose=0,
)
search.fit(X, y)

print(f"best CV R2: {search.best_score_:.4f}")
print("best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

joblib.dump(search.best_estimator_, MODEL_OUT)
print()
print(f"saved: {MODEL_OUT}")
