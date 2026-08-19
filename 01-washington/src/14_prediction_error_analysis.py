"""
최종 모델(XGBoost 튜닝)로 731일 전체를 예측하고, 실제값과의 차이가 가장 큰 10일을 찾는다.
그 10일이 13_isolation_forest_outliers.py가 찾은 이상치 37일과 얼마나 겹치는지도 함께 확인한다.

주의: 모델은 80%(584일)로 학습되었으므로 731일 전체를 예측하면 학습에 쓰인 날의 오차는
낙관적으로 나온다. 그래서 결과표에 각 날이 train/test 중 어디였는지 함께 표시한다.

입력: data/day_processed.csv, models/xgboost_tuned_final.joblib,
      tables/isolation_forest_outliers.csv
출력: tables/worst10_predictions.csv
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

DATA_PATH = "data/day_processed.csv"
MODEL_PATH = "models/xgboost_tuned_final.joblib"
OUTLIERS_PATH = "tables/isolation_forest_outliers.csv"
OUT_TABLE = "tables/worst10_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
                  "weathersit", "temp", "atemp", "hum", "windspeed"]


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    split_tag = pd.Series("test", index=df.index)
    split_tag.loc[train_df.index] = "train"

    pred = model.predict(df[BASE_FEATURES])

    result = pd.DataFrame({
        "date": df["dteday"],
        "actual": df["cnt"],
        "predicted": np.round(pred).astype(int),
        "diff": df["cnt"] - np.round(pred).astype(int),
        "split": split_tag.values,
    })
    result["abs_diff"] = result["diff"].abs()
    top10 = result.sort_values("abs_diff", ascending=False).head(10).drop(columns="abs_diff")

    outlier_dates = set(pd.read_csv(OUTLIERS_PATH)["dteday"])
    top10["is_isoforest_outlier"] = top10["date"].isin(outlier_dates)

    print(top10.to_string(index=False))
    print()
    overlap = top10[top10["is_isoforest_outlier"]]["date"].tolist()
    print(f"이상치 목록과 겹치는 날: {overlap}")

    top10.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    print(f"saved: {OUT_TABLE}")


if __name__ == "__main__":
    main()
