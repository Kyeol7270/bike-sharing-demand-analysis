"""2016 예측 오차 분석. ANALYSIS_PLAN.md 3-9절.
최종 모델(2016 홀드아웃 R2 기준 최고)로 2016 전체를 예측해 오차 상위 10일을 추출하고,
2016에 대해 별도로 돌린 Isolation Forest 이상치 목록과 교집합을 확인한다.
(2015 이상치 목록과는 대상 기간이 달라 여기서 2016용 이상치를 새로 계산한다.)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

TRAIN_PATH = "data/day_2015.csv"
TEST_PATH = "data/day_2016.csv"
COMPARISON_TABLE = "tables/final_model_comparison.csv"
MODEL_DIR = "models"
ERROR_TABLE_OUT = "tables/worst10_predictions_2016.csv"
OUTLIER_TABLE_OUT = "tables/isolation_forest_outliers_2016.csv"
RANDOM_STATE = 42
CONTAMINATION = 0.05

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)
test["weekday_ord"] = test["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
TREE_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                  "weather_code", "t1", "t2", "hum", "wind_speed"]
LINEAR_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                    "weather_code", "t1", "hum", "wind_speed"]
ONEHOT_CAT_COLS = ["season", "weather_code", "weekday"]
ONEHOT_NUM_COLS = ["month", "day", "is_holiday", "is_weekend", "t1", "hum", "wind_speed"]

comparison = pd.read_csv(COMPARISON_TABLE).sort_values("holdout_R2_2016", ascending=False)
best_name = comparison.iloc[0]["model"]
print(f"최고 모델(2016 홀드아웃 R2 기준): {best_name}")

model = joblib.load(f"{MODEL_DIR}/{best_name}_final.joblib")
is_tree = best_name.startswith("RandomForest") or best_name.startswith("XGBoost")

if is_tree:
    X_test = test[TREE_FEATURES]
elif best_name == "LinearRegression_ordinal":
    X_test = test[LINEAR_FEATURES]
elif best_name == "LinearRegression_onehot":
    # 10_final_model_comparison.py와 동일한 방식: train+test를 합쳐서 인코딩해야
    # weather_code=26(train엔 없고 test에만 있는 범주)도 컬럼이 어긋나지 않음
    combined = pd.concat([train, test], keys=["train", "test"])
    combined_onehot = pd.get_dummies(combined[ONEHOT_CAT_COLS], columns=ONEHOT_CAT_COLS, drop_first=True)
    X_onehot = pd.concat([combined[ONEHOT_NUM_COLS], combined_onehot], axis=1)
    X_test = X_onehot.loc["test"]
else:
    raise SystemExit(f"알 수 없는 모델 이름: {best_name}")

pred = model.predict(X_test)
error = np.abs(test[TARGET].values - pred)

error_df = pd.DataFrame({
    "date": test["year"].astype(str) + "-" + test["month"].astype(str).str.zfill(2) + "-" + test["day"].astype(str).str.zfill(2),
    "actual": test[TARGET].values,
    "predicted": pred,
    "abs_error": error,
})
worst10 = error_df.sort_values("abs_error", ascending=False).head(10)
print()
print("=== 2016 예측 오차 상위 10일 ===")
print(worst10.to_string(index=False))
worst10.to_csv(ERROR_TABLE_OUT, index=False)

# 2016 이상치 탐지 (train=2015에서 학습한 것이 아니라 2016 자체 분포 기준으로 별도 탐지)
OUTLIER_FEATURES = ["t1", "t2", "hum", "wind_speed", "weather_code", "cnt"]
X_iso = test[OUTLIER_FEATURES]
iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
iso.fit(X_iso)
is_outlier = iso.predict(X_iso) == -1
outlier_dates = set(error_df.loc[is_outlier, "date"])
print()
print(f"2016 이상치 {is_outlier.sum()}건 / {len(test)}일")

pd.DataFrame({"date": error_df.loc[is_outlier, "date"]}).to_csv(OUTLIER_TABLE_OUT, index=False)

worst10_dates = set(worst10["date"])
overlap = worst10_dates & outlier_dates
print()
print(f"오차 상위 10일 중 이상치와 겹치는 날: {len(overlap)}건 -> {sorted(overlap)}")
print(f"saved: {ERROR_TABLE_OUT}, {OUTLIER_TABLE_OUT}")
