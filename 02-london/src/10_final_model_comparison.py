"""최종 모델 비교: 2015 전체로 학습 -> 2016 전체(day_2016.csv, 진짜 홀드아웃) 예측.
ANALYSIS_PLAN.md 3-6절. 하드코딩 없이 원본에서 매번 재계산한다(워싱턴 11번 스크립트가
하드코딩 문제로 재작성됐던 사례를 재발 방지 차원에서 처음부터 반영).

튜닝된 RandomForest/XGBoost는 08/09번 스크립트가 저장한 models/*.joblib을 그대로 불러와
2015 전체로 재학습한다(튜닝 시점엔 CV 내부에서만 학습됐으므로 최종 예측 전에 전체 재학습 필요).
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TRAIN_PATH = "data/day_2015.csv"
TEST_PATH = "data/day_2016.csv"
TABLE_OUT = "tables/final_model_comparison.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)
test["weekday_ord"] = test["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
y_train, y_test = train[TARGET], test[TARGET]

LINEAR_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                    "weather_code", "t1", "hum", "wind_speed"]
TREE_FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
                  "weather_code", "t1", "t2", "hum", "wind_speed"]
ONEHOT_CAT_COLS = ["season", "weather_code", "weekday"]
ONEHOT_NUM_COLS = ["month", "day", "is_holiday", "is_weekend", "t1", "hum", "wind_speed"]

X_train_linear, X_test_linear = train[LINEAR_FEATURES], test[LINEAR_FEATURES]
X_train_tree, X_test_tree = train[TREE_FEATURES], test[TREE_FEATURES]

# 원-핫: train+test를 합쳐서 인코딩 후 다시 분리 -> weather_code=26(눈, train엔 없고 test에만 1건)
# 처럼 한쪽에만 있는 범주도 두 세트의 컬럼이 항상 일치하도록 보장
combined = pd.concat([train, test], keys=["train", "test"])
combined_onehot = pd.get_dummies(combined[ONEHOT_CAT_COLS], columns=ONEHOT_CAT_COLS, drop_first=True)
X_onehot = pd.concat([combined[ONEHOT_NUM_COLS], combined_onehot], axis=1)
X_train_onehot = X_onehot.loc["train"]
X_test_onehot = X_onehot.loc["test"]

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []


def evaluate(name, model, X_train, X_test, cv_X=None):
    cv_source = cv_X if cv_X is not None else X_train
    cv_r2 = cross_val_score(model, cv_source, y_train, cv=kf, scoring="r2")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)

    results.append({
        "model": name,
        "CV_R2_2015_mean": cv_r2.mean(), "CV_R2_2015_std": cv_r2.std(),
        "holdout_R2_2016": r2, "holdout_RMSE_2016": rmse, "holdout_MAE_2016": mae,
    })
    print(f"[{name}] CV(2015) R2={cv_r2.mean():.4f}+/-{cv_r2.std():.4f}  "
          f"holdout(2016) R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    joblib.dump(model, f"{MODEL_DIR}/{name}_final.joblib")
    return pred


evaluate("LinearRegression_ordinal", LinearRegression(), X_train_linear, X_test_linear)
evaluate("LinearRegression_onehot", LinearRegression(), X_train_onehot, X_test_onehot)

rf_tuned = joblib.load(f"{MODEL_DIR}/random_forest_tuned.joblib")
evaluate("RandomForest_tuned", rf_tuned, X_train_tree, X_test_tree)
evaluate("RandomForest_default", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
         X_train_tree, X_test_tree)

xgb_tuned = joblib.load(f"{MODEL_DIR}/xgboost_tuned.joblib")
evaluate("XGBoost_tuned", xgb_tuned, X_train_tree, X_test_tree)
evaluate("XGBoost_default", XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
         X_train_tree, X_test_tree)

results_df = pd.DataFrame(results).sort_values("holdout_R2_2016", ascending=False)
print()
print(results_df.to_string(index=False))
results_df.to_csv(TABLE_OUT, index=False)
print()
print(f"saved: {TABLE_OUT}")
