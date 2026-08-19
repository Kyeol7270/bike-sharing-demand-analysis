"""회귀 모델 비교: LinearRegression(순서형) vs LinearRegression(원-핫) vs RandomForest vs XGBoost.
ANALYSIS_PLAN.md 3-4절. 평가는 train(2015) 내부 5-fold CV로만 진행 (2016은 건드리지 않음).

- 선형회귀는 t1-t2 다중공선성(VIF 88~91) 때문에 t2 제외, t1만 사용.
- 트리 모델(RandomForest/XGBoost)은 t1/t2 둘 다 사용, weather_code/season 원본 값 그대로,
  weekday만 달력 순서 정수로 인코딩.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

TRAIN_PATH = "data/day_2015.csv"
TABLE_OUT = "tables/model_comparison.csv"
RANDOM_STATE = 42

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
y = train[TARGET]

LINEAR_ORDINAL_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday",
                            "is_weekend", "weather_code", "t1", "hum", "wind_speed"]
TREE_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                  "weather_code", "t1", "t2", "hum", "wind_speed"]
ONEHOT_CAT_COLS = ["season", "weather_code", "weekday"]
ONEHOT_NUM_COLS = ["month", "day", "is_holiday", "is_weekend", "t1", "hum", "wind_speed"]

X_linear = train[LINEAR_ORDINAL_FEATURES]
X_tree = train[TREE_FEATURES]

train_onehot = pd.get_dummies(train[ONEHOT_CAT_COLS], columns=ONEHOT_CAT_COLS, drop_first=True)
X_onehot = pd.concat([train[ONEHOT_NUM_COLS], train_onehot], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []


def evaluate(name, model, X):
    r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
    mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse = -cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    results.append({
        "model": name, "n_features": X.shape[1],
        "CV_R2_mean": r2.mean(), "CV_R2_std": r2.std(),
        "CV_MAE_mean": mae.mean(), "CV_RMSE_mean": rmse.mean(),
    })
    print(f"[{name}] features={X.shape[1]}  "
          f"CV R2={r2.mean():.4f}+/-{r2.std():.4f}  "
          f"CV MAE={mae.mean():.2f}  CV RMSE={rmse.mean():.2f}")


evaluate("LinearRegression_ordinal", LinearRegression(), X_linear)
evaluate("LinearRegression_onehot", LinearRegression(), X_onehot)
evaluate("RandomForest_default", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE), X_tree)
evaluate("XGBoost_default", XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0), X_tree)

results_df = pd.DataFrame(results).sort_values("CV_R2_mean", ascending=False)
print()
print(results_df.to_string(index=False))
results_df.to_csv(TABLE_OUT, index=False)
print()
print(f"saved: {TABLE_OUT}")
