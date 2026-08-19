"""데이터 개요 확인: data/day_2015.csv(train) / data/day_2016.csv(test).
ANALYSIS_PLAN.md 3-1절: 전처리 상태 재확인 + train/test 분포 쏠림 확인.
"""
import pandas as pd

TRAIN_PATH = "data/day_2015.csv"
TEST_PATH = "data/day_2016.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"train: {train.shape}, test: {test.shape}")
print()
print("=== 결측치 ===")
print("train:", train.isnull().sum().sum(), " test:", test.isnull().sum().sum())
print()

print("=== n_hours_observed < 24인 날 (저신뢰) ===")
print("train:", (train["n_hours_observed"] < 24).sum(), "/", len(train))
print("test :", (test["n_hours_observed"] < 24).sum(), "/", len(test))
print()

print("=== season 분포 (train vs test) ===")
print(pd.concat([
    train["season"].value_counts().sort_index().rename("train"),
    test["season"].value_counts().sort_index().rename("test"),
], axis=1))
print()

print("=== is_holiday / is_weekend 분포 ===")
for c in ["is_holiday", "is_weekend"]:
    print(c)
    print(pd.concat([
        train[c].value_counts().sort_index().rename("train"),
        test[c].value_counts().sort_index().rename("test"),
    ], axis=1))
    print()

print("=== weather_code 분포 (train vs test) ===")
print(pd.concat([
    train["weather_code"].value_counts().sort_index().rename("train"),
    test["weather_code"].value_counts().sort_index().rename("test"),
], axis=1))
print()

print("=== cnt 기술통계 ===")
print(pd.concat([
    train["cnt"].describe().rename("train"),
    test["cnt"].describe().rename("test"),
], axis=1))
