"""6개 회귀모델의 5-fold CV 성능(R2/RMSE/MAE) 비교 시각화.
수치는 이전 단계들(compare_regression_models.py, validate_models.py,
tune_xgboost.py, tune_random_forest.py)에서 이미 계산된 CV 결과를 취합.
결과는 figures/, tables/ 에 저장.
"""
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"   # 최고 성능 모델 강조색
GRAY = "#c3c2b7"     # 나머지 비강조

OUT_FIG = "figures/model_performance_comparison.png"
OUT_TABLE = "tables/model_comparison_final.csv"

data = [
    {"model": "LinearRegression\n(순서형)", "CV_R2": 0.783, "CV_RMSE": 887.61, "CV_MAE": 656.23},
    {"model": "LinearRegression\n(원-핫)", "CV_R2": 0.821, "CV_RMSE": 805.42, "CV_MAE": 586.77},
    {"model": "RandomForest\n(default)", "CV_R2": 0.875, "CV_RMSE": 673.24, "CV_MAE": 457.42},
    {"model": "RandomForest\n(튜닝)", "CV_R2": 0.874, "CV_RMSE": 676.85, "CV_MAE": 465.15},
    {"model": "XGBoost\n(default)", "CV_R2": 0.872, "CV_RMSE": 681.54, "CV_MAE": 474.62},
    {"model": "XGBoost\n(튜닝)", "CV_R2": 0.893, "CV_RMSE": 621.66, "CV_MAE": 439.36},
]
df = pd.DataFrame(data)
df.to_csv(OUT_TABLE, index=False)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)


def panel(ax, metric, title, higher_is_better):
    sorted_df = df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    best_val = sorted_df[metric].iloc[0]
    colors = [ACCENT if v == best_val else GRAY for v in sorted_df[metric]]

    y_pos = range(len(sorted_df))
    bars = ax.barh(y_pos, sorted_df[metric], color=colors, height=0.6, zorder=2)
    ax.invert_yaxis()  # 1위가 맨 위로
    style_ax(ax)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sorted_df["model"], fontsize=9, color=INK_SECONDARY)

    xmax = sorted_df[metric].max()
    for bar, v, is_best in zip(bars, sorted_df[metric], sorted_df[metric] == best_val):
        label = f"{v:.3f}" if metric == "CV_R2" else f"{v:,.0f}"
        color = ACCENT if is_best else INK_SECONDARY
        weight = "bold" if is_best else "normal"
        ax.text(v + xmax * 0.02, bar.get_y() + bar.get_height() / 2, label,
                 va="center", fontsize=9, color=color, fontweight=weight)

    ax.set_xlim(0, xmax * 1.18)
    ax.set_title(title, fontsize=11, color=INK_PRIMARY, pad=10)


fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), dpi=150, facecolor=SURFACE)
fig.suptitle("회귀모델 성능 비교 (5-fold 교차검증 기준, cnt 예측)", fontsize=13,
             color=INK_PRIMARY, y=1.02)

panel(axes[0], "CV_R2", "R² (높을수록 좋음)", higher_is_better=True)
panel(axes[1], "CV_RMSE", "RMSE (낮을수록 좋음)", higher_is_better=False)
panel(axes[2], "CV_MAE", "MAE (낮을수록 좋음)", higher_is_better=False)

fig.text(0.5, -0.02, "파란색 = 각 지표별 최고 성능 모델", ha="center", fontsize=9, color=INK_MUTED)

fig.tight_layout()
fig.savefig(OUT_FIG, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print(f"saved: {OUT_FIG}")
print(f"saved: {OUT_TABLE}")
