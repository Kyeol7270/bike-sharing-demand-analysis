"""과적합 검증. ANALYSIS_PLAN.md 3-5절.

train(2015)이 362행뿐이라 이 안에서 또 홀드아웃을 떼어내는 대신, 5-fold CV의 각 fold에서
train-fold R²와 validation-fold R²를 직접 비교해 과적합 정도(gap)를 확인한다(홀드아웃 2016은
여전히 손대지 않음).
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

TRAIN_PATH = "data/day_2015.csv"
TABLE_OUT = "tables/cv_overfitting_check.csv"
RANDOM_STATE = 42

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
y = train[TARGET]

TREE_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                  "weather_code", "t1", "t2", "hum", "wind_speed"]
LINEAR_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                    "weather_code", "t1", "hum", "wind_speed"]

X_tree = train[TREE_FEATURES]
X_linear = train[LINEAR_FEATURES]

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

MODELS = {
    "LinearRegression_ordinal": (LinearRegression(), X_linear),
    "RandomForest_default": (RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE), X_tree),
    "XGBoost_default": (XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0), X_tree),
}

rows = []
for name, (model, X) in MODELS.items():
    train_r2s, val_r2s = [], []
    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        train_r2s.append(r2_score(y_tr, model.predict(X_tr)))
        val_r2s.append(r2_score(y_val, model.predict(X_val)))
    train_r2 = np.mean(train_r2s)
    val_r2 = np.mean(val_r2s)
    gap = train_r2 - val_r2
    rows.append({"model": name, "train_R2_mean": train_r2, "val_R2_mean": val_r2, "gap": gap})
    print(f"[{name}] train R2={train_r2:.4f}  val R2={val_r2:.4f}  gap={gap:.4f}")

result_df = pd.DataFrame(rows)
result_df.to_csv(TABLE_OUT, index=False)
print()
print(f"saved: {TABLE_OUT}")
