"""변수 중요도 분석. ANALYSIS_PLAN.md 3-7절.
tables/final_model_comparison.csv에서 2016 홀드아웃 R2 기준 최고 모델을 선택해
회귀 계수 + permutation importance(2016 홀드아웃 R2 하락폭)를 교차 확인한다.

최종 채택 모델이 원-핫 인코딩 선형회귀(LinearRegression_onehot)로 확인됨에 따라,
원-핫으로 쪼개진 범주형 컬럼(season/weather_code/weekday)은 같은 원본 변수에 속한 더미
컬럼을 한번에 함께 셔플하는 "그룹 permutation importance"로 측정한다(개별 더미 컬럼만
따로 섞으면 원본 변수의 진짜 영향력을 과소평가하게 됨).
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TRAIN_PATH = "data/day_2015.csv"
TEST_PATH = "data/day_2016.csv"
COMPARISON_TABLE = "tables/final_model_comparison.csv"
MODEL_DIR = "models"
FIG_OUT = "figures/feature_importance.png"
TABLE_OUT = "tables/feature_importance.csv"
RANDOM_STATE = 42
N_REPEATS = 30

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)
test["weekday_ord"] = test["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
y_test = test[TARGET]

ONEHOT_CAT_COLS = ["season", "weather_code", "weekday"]
ONEHOT_NUM_COLS = ["month", "day", "is_holiday", "is_weekend", "t1", "hum", "wind_speed"]

comparison = pd.read_csv(COMPARISON_TABLE).sort_values("holdout_R2_2016", ascending=False)
best_name = comparison.iloc[0]["model"]
print(f"최고 모델(2016 홀드아웃 R2 기준): {best_name}")
assert best_name == "LinearRegression_onehot", (
    f"이 스크립트는 LinearRegression_onehot 전용으로 작성됨 (실제 최고 모델: {best_name}) "
    f"— 다른 모델이 1위면 스크립트를 그 모델에 맞게 다시 작성해야 함"
)

model = joblib.load(f"{MODEL_DIR}/{best_name}_final.joblib")

combined = pd.concat([train, test], keys=["train", "test"])
combined_onehot = pd.get_dummies(combined[ONEHOT_CAT_COLS], columns=ONEHOT_CAT_COLS, drop_first=True)
X_onehot = pd.concat([combined[ONEHOT_NUM_COLS], combined_onehot], axis=1)
X_test = X_onehot.loc["test"]

# 원본 변수 -> 그 변수에 해당하는 (원-핫 포함) 컬럼 목록
groups = {c: [c] for c in ONEHOT_NUM_COLS}
for cat in ONEHOT_CAT_COLS:
    groups[cat] = [c for c in combined_onehot.columns if c.startswith(f"{cat}_")]

print()
print("=== 회귀 계수 (원-핫 더미 컬럼별) ===")
coef_df = pd.DataFrame({"feature": X_test.columns, "coef": model.coef_})
print(coef_df.to_string(index=False))

baseline_r2 = r2_score(y_test, model.predict(X_test))
rng = np.random.default_rng(RANDOM_STATE)

rows = []
for var, cols in groups.items():
    drops = []
    for _ in range(N_REPEATS):
        perm_idx = rng.permutation(len(X_test))
        X_shuffled = X_test.copy()
        X_shuffled[cols] = X_test[cols].to_numpy()[perm_idx]
        r2 = r2_score(y_test, model.predict(X_shuffled))
        drops.append(baseline_r2 - r2)
    rows.append({"variable": var, "n_dummy_cols": len(cols),
                  "permutation_importance_mean": np.mean(drops), "std": np.std(drops)})

imp_df = pd.DataFrame(rows).sort_values("permutation_importance_mean", ascending=False)
print()
print("=== 그룹 permutation importance (2016 홀드아웃 R2 하락폭, 변수 단위) ===")
print(imp_df.to_string(index=False))
imp_df.to_csv(TABLE_OUT, index=False)

plt.figure(figsize=(7, 5))
imp_df.set_index("variable")["permutation_importance_mean"].sort_values().plot.barh(color="#5b8fc7")
plt.title(f"{best_name} — 변수별 permutation importance (2016 홀드아웃)")
plt.xlabel("R2 하락폭")
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
plt.close()

print()
print(f"saved: {TABLE_OUT}, {FIG_OUT}")
