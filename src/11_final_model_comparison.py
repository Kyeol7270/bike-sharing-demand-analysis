"""6개 회귀모델의 성능 비교 — 5-fold 교차검증과 홀드아웃 두 프로토콜을 나란히 제시.

두 프로토콜을 함께 보여주는 이유:
  - 교차검증(CV) 점수는 데이터 전체를 쓰므로 표본 변동에 덜 흔들리지만, 튜닝 모델의 경우
    하이퍼파라미터를 고른 데이터와 점수를 잰 데이터가 겹쳐 **낙관적으로 편향**된다.
  - 홀드아웃(test) 점수는 튜닝에 쓰이지 않은 20%에서 측정하므로 그 편향이 없다.
  두 결과가 어긋나는 지표(MAE)가 실제로 존재하므로, 한쪽만 싣으면 결론이 왜곡된다.

수치는 하드코딩하지 않고 data/day_processed.csv에서 매번 다시 계산한다.
튜닝된 하이퍼파라미터는 09_tune_xgboost.py / 10_tune_random_forest.py의 탐색 결과
(tables/*_tuning_results.csv의 best_params)를 그대로 옮긴 값이다.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"   # 최고 성능 모델 강조색
GRAY = "#c3c2b7"     # 나머지 비강조

DATA_PATH = "data/day_processed.csv"
OUT_FIG = "figures/model_performance_comparison.png"
OUT_TABLE = "tables/model_comparison_final.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

TARGET = "cnt"
BASE_FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
                  "weathersit", "temp", "atemp", "hum", "windspeed"]
CAT_FOR_ONEHOT = ["season", "mnth", "weekday", "weathersit"]

df = pd.read_csv(DATA_PATH)

df_onehot = pd.get_dummies(df, columns=CAT_FOR_ONEHOT, drop_first=True)
onehot_features = [c for c in df_onehot.columns
                    if c not in ["instant", "dteday", "casual", "registered", TARGET]]

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
y_train, y_test = train_df[TARGET], test_df[TARGET]
y_full = df[TARGET]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# n_estimators=300은 명시적으로 지정한 값이다("완전 default"가 아님 — 라벨에도 그대로 표기).
MODELS = [
    ("LinearRegression\n(순서형)",
     lambda: LinearRegression(), df[BASE_FEATURES], False),
    ("LinearRegression\n(원-핫)",
     lambda: LinearRegression(), df_onehot[onehot_features], False),
    ("RandomForest\n(n_est=300)",
     lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
     df[BASE_FEATURES], False),
    ("RandomForest\n(튜닝)",
     lambda: RandomForestRegressor(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                                    max_features=0.5, max_depth=15, random_state=RANDOM_STATE),
     df[BASE_FEATURES], True),
    ("XGBoost\n(n_est=300)",
     lambda: XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
     df[BASE_FEATURES], False),
    ("XGBoost\n(튜닝)",
     lambda: XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=1, reg_alpha=1, reg_lambda=3,
                           random_state=RANDOM_STATE, verbosity=0),
     df[BASE_FEATURES], True),
]

rows = []
for name, make_model, X_full, is_tuned in MODELS:
    # --- 5-fold CV (전체 데이터) ---
    cv_r2 = cross_val_score(make_model(), X_full, y_full, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(make_model(), X_full, y_full, cv=kf,
                                scoring="neg_root_mean_squared_error")
    cv_mae = -cross_val_score(make_model(), X_full, y_full, cv=kf,
                               scoring="neg_mean_absolute_error")

    # --- 홀드아웃 (train으로 학습, test 20%로 평가) ---
    model = make_model()
    model.fit(X_full.loc[train_df.index], y_train)
    pred = model.predict(X_full.loc[test_df.index])

    rows.append({
        "model": name.replace("\n", " "),
        "tuned": is_tuned,
        "CV_R2": cv_r2.mean(), "CV_R2_std": cv_r2.std(),
        "CV_RMSE": cv_rmse.mean(), "CV_RMSE_std": cv_rmse.std(),
        "CV_MAE": cv_mae.mean(), "CV_MAE_std": cv_mae.std(),
        "holdout_R2": r2_score(y_test, pred),
        "holdout_RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "holdout_MAE": mean_absolute_error(y_test, pred),
    })
    print(f"[{name.replace(chr(10), ' ')}]")
    print(f"  CV      R2={cv_r2.mean():.3f}+/-{cv_r2.std():.3f}  "
          f"RMSE={cv_rmse.mean():.2f}  MAE={cv_mae.mean():.2f}")
    print(f"  holdout R2={rows[-1]['holdout_R2']:.3f}  "
          f"RMSE={rows[-1]['holdout_RMSE']:.2f}  MAE={rows[-1]['holdout_MAE']:.2f}")

res = pd.DataFrame(rows)
res.to_csv(OUT_TABLE, index=False)

plot_df = res.copy()
plot_df["label"] = [m[0] for m in MODELS]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)


def panel(ax, metric, title, higher_is_better, std_col=None):
    sorted_df = plot_df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    best_val = sorted_df[metric].iloc[0]
    colors = [ACCENT if v == best_val else GRAY for v in sorted_df[metric]]
    errs = sorted_df[std_col].values if std_col else np.zeros(len(sorted_df))

    y_pos = range(len(sorted_df))
    bars = ax.barh(y_pos, sorted_df[metric], xerr=errs if std_col else None,
                    color=colors, height=0.6, zorder=2,
                    error_kw=dict(ecolor=INK_MUTED, elinewidth=1, capsize=3))
    ax.invert_yaxis()  # 1위가 맨 위로
    style_ax(ax)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_df["label"], fontsize=8.5, color=INK_SECONDARY)

    xmax = (sorted_df[metric].values + errs).max()
    for bar, v, e, is_best in zip(bars, sorted_df[metric], errs,
                                   sorted_df[metric] == best_val):
        label = f"{v:.3f}" if "R2" in metric else f"{v:,.0f}"
        ax.text(v + e + xmax * 0.03, bar.get_y() + bar.get_height() / 2, label,
                 va="center", fontsize=8.5, color=ACCENT if is_best else INK_SECONDARY,
                 fontweight="bold" if is_best else "normal")

    ax.set_xlim(0, xmax * 1.22)
    ax.set_title(title, fontsize=10.5, color=INK_PRIMARY, pad=8)


fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150, facecolor=SURFACE)
fig.suptitle("회귀모델 성능 비교 — 교차검증 vs 홀드아웃 (cnt 예측)", fontsize=13,
             color=INK_PRIMARY, y=0.99)

panel(axes[0][0], "CV_R2", "① 5-fold CV · R² (높을수록 좋음)", True, "CV_R2_std")
panel(axes[0][1], "CV_RMSE", "① 5-fold CV · RMSE (낮을수록 좋음)", False, "CV_RMSE_std")
panel(axes[0][2], "CV_MAE", "① 5-fold CV · MAE (낮을수록 좋음)", False, "CV_MAE_std")
panel(axes[1][0], "holdout_R2", "② 홀드아웃 20% · R² (높을수록 좋음)", True)
panel(axes[1][1], "holdout_RMSE", "② 홀드아웃 20% · RMSE (낮을수록 좋음)", False)
panel(axes[1][2], "holdout_MAE", "② 홀드아웃 20% · MAE (낮을수록 좋음)", False)

fig.text(0.5, 0.015,
         "파란색 = 각 지표별 1위    |    오차막대 = 5-fold 표준편차\n"
         "※ 상위 모델 간 CV R² 차이는 fold 표준편차(±0.02~0.03)보다 작아 통계적으로 확정적이지 않다.\n"
         "※ 튜닝 모델의 CV 점수는 하이퍼파라미터를 고른 데이터와 겹치는 구간에서 측정돼 낙관적으로 편향돼 있다 — "
         "편향 없는 비교는 아래 홀드아웃 행이며, 여기서 MAE 1위는 RandomForest다.",
         ha="center", fontsize=8.5, color=INK_MUTED, linespacing=1.6)

fig.tight_layout(rect=[0, 0.075, 1, 0.98])
fig.savefig(OUT_FIG, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print()
print(res.to_string(index=False))
print()
print(f"saved: {OUT_FIG}")
print(f"saved: {OUT_TABLE}")
