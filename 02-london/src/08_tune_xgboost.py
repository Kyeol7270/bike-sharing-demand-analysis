"""XGBoost 하이퍼파라미터 튜닝. ANALYSIS_PLAN.md 3-5절.
RandomizedSearchCV(5-fold CV, n_iter=60), train(2015) 내부에서만 진행 (2016 미사용).
n_jobs=1 고정 — 경로에 한글(사용자명)이 포함돼 있어 n_jobs=-1 사용 시 joblib
멀티프로세싱이 UnicodeEncodeError로 실패하는 환경 이슈가 있음(워싱턴 프로젝트에서 확인된 이슈).
"""
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold

TRAIN_PATH = "data/day_2015.csv"
MODEL_OUT = "models/xgboost_tuned.joblib"
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
    "max_depth": [2, 3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [0.5, 1, 1.5, 2],
}

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
base_model = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)

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
