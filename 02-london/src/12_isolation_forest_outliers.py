"""이상치 탐지 (train=2015). ANALYSIS_PLAN.md 3-8절.
Isolation Forest(contamination=5%)로 이상치 탐지 후, 중앙값 치환(median ablation)으로
각 이상치의 원인 변수를 특정한다 (워싱턴 13번 스크립트와 동일한 방법론).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

TRAIN_PATH = "data/day_2015.csv"
TABLE_OUT = "tables/isolation_forest_outliers_2015.csv"
RANDOM_STATE = 42
CONTAMINATION = 0.05

FEATURES = ["t1", "t2", "hum", "wind_speed", "weather_code", "cnt"]

train = pd.read_csv(TRAIN_PATH)
X = train[FEATURES]

iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
iso.fit(X)
scores = iso.decision_function(X)
is_outlier = iso.predict(X) == -1

print(f"이상치 {is_outlier.sum()}건 / {len(train)}일 ({is_outlier.sum() / len(train) * 100:.1f}%)")

medians = X.median()
rows = []
for idx in train.index[is_outlier]:
    base_score = scores[idx]
    recover = {}
    for col in FEATURES:
        x_mod = X.loc[[idx]].copy()
        x_mod[col] = medians[col]
        new_score = iso.decision_function(x_mod)[0]
        recover[col] = new_score - base_score
    cause = max(recover, key=recover.get)
    rows.append({
        "date": f"{train.loc[idx, 'year']}-{train.loc[idx, 'month']:02d}-{train.loc[idx, 'day']:02d}",
        "score": base_score,
        "cause": cause,
        "cnt": train.loc[idx, "cnt"],
        **{f"recover_{c}": recover[c] for c in FEATURES},
    })

result_df = pd.DataFrame(rows).sort_values("score")
print()
print(result_df[["date", "score", "cause", "cnt"]].to_string(index=False))

print()
print("=== 원인 변수 분포 ===")
print(result_df["cause"].value_counts())

result_df.to_csv(TABLE_OUT, index=False)
print()
print(f"saved: {TABLE_OUT}")
