"""최종 채택 모델(XGBoost 튜닝)의 변수 중요도 분석.
1) 모델 내장 importance (gain 기준)
2) permutation importance (test셋에서 변수를 무작위로 섞었을 때 성능 하락폭 -> 더 신뢰도 높은 지표)
데이터: data/day_processed.csv, 타겟: cnt (casual/registered 제외)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

DATA_PATH = "data/day_processed.csv"
MODEL_PATH = "models/xgboost_tuned_final.joblib"
OUT_FIG = "figures/feature_importance_xgboost.png"
OUT_TABLE = "tables/feature_importance_xgboost.csv"

FEATURE_LABELS = {
    "season": "season (계절)", "yr": "yr (연도)", "mnth": "mnth (월)",
    "holiday": "holiday (공휴일)", "weekday": "weekday (요일)",
    "workingday": "workingday (평일여부)", "weathersit": "weathersit (날씨등급)",
    "temp": "temp (기온)", "atemp": "atemp (체감기온)", "hum": "hum (습도)",
    "windspeed": "windspeed (풍속)",
}
BASE_FEATURES = list(FEATURE_LABELS.keys())
TARGET = "cnt"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
X = df[BASE_FEATURES]
y = df[TARGET]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

model = joblib.load(MODEL_PATH)

# 1. 모델 내장 importance (gain 기준)
gain_imp = pd.Series(model.feature_importances_, index=BASE_FEATURES).sort_values(ascending=False)

# 2. permutation importance (test셋 기준)
perm = permutation_importance(model, X_test, y_test, n_repeats=30,
                               random_state=RANDOM_STATE, scoring="r2")
perm_imp = pd.Series(perm.importances_mean, index=BASE_FEATURES).sort_values(ascending=False)

result = pd.DataFrame({
    "gain_importance": gain_imp.reindex(BASE_FEATURES),
    "permutation_importance_mean_R2_drop": pd.Series(perm.importances_mean, index=BASE_FEATURES),
    "permutation_importance_std": pd.Series(perm.importances_std, index=BASE_FEATURES),
})
result = result.sort_values("permutation_importance_mean_R2_drop", ascending=False)
result.to_csv(OUT_TABLE)

print("=== Gain 기준 (모델 내장) ===")
print(gain_imp.to_string())
print()
print("=== Permutation Importance 기준 (test셋 R2 하락폭) ===")
print(perm_imp.to_string())

# --- 시각화: permutation importance 기준 정렬 막대그래프 ---
plot_df = result.sort_values("permutation_importance_mean_R2_drop", ascending=True)
labels = [FEATURE_LABELS[f] for f in plot_df.index]
values = plot_df["permutation_importance_mean_R2_drop"].values
errs = plot_df["permutation_importance_std"].values

n = len(values)
ramp_idx = np.linspace(0, len(SEQ_RAMP) - 1, n).astype(int)
colors = [SEQ_RAMP[i] for i in ramp_idx]

fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
bars = ax.barh(labels, values, xerr=errs, color=colors, height=0.6,
                error_kw=dict(ecolor=INK_MUTED, elinewidth=1, capsize=3), zorder=2)
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK_SECONDARY)
ax.set_xlabel("Permutation Importance (R² 하락폭)", fontsize=10, color=INK_SECONDARY)
ax.set_title("XGBoost(튜닝) 변수 중요도 — cnt 예측 기여도", fontsize=12, color=INK_PRIMARY, pad=12)

for bar, v in zip(bars, values):
    ax.text(v + max(values) * 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
            va="center", fontsize=8.5, color=INK_SECONDARY)

fig.tight_layout()
fig.savefig(OUT_FIG, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print()
print(f"saved: {OUT_FIG}")
print(f"saved: {OUT_TABLE}")
