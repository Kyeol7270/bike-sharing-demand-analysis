"""최종 모델 성능 비교 시각화 (CV vs 2016 홀드아웃). README에 삽입하는 대표 그림."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TABLE = "tables/final_model_comparison.csv"
FIG_OUT = "figures/model_performance_comparison.png"

df = pd.read_csv(TABLE).sort_values("holdout_R2_2016", ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(2, 1, figsize=(9, 8))

x = np.arange(len(df))
axes[0].bar(x - 0.2, df["CV_R2_2015_mean"], width=0.4, yerr=df["CV_R2_2015_std"],
            label="CV R² (2015 내부, 5-fold)", color="#9aa0c8", capsize=4)
axes[0].bar(x + 0.2, df["holdout_R2_2016"], width=0.4,
            label="홀드아웃 R² (2016 예측)", color="#d7726b")
axes[0].set_xticks(x)
axes[0].set_xticklabels(df["model"], rotation=20, ha="right")
axes[0].set_ylabel("R²")
axes[0].set_title("모델별 R²: 2015 내부 CV vs 2016 홀드아웃 예측")
axes[0].legend()
axes[0].set_ylim(0, 0.9)

axes[1].bar(x - 0.2, df["holdout_RMSE_2016"], width=0.4, label="RMSE (2016)", color="#5b8fc7")
axes[1].bar(x + 0.2, df["holdout_MAE_2016"], width=0.4, label="MAE (2016)", color="#e8a33d")
axes[1].set_xticks(x)
axes[1].set_xticklabels(df["model"], rotation=20, ha="right")
axes[1].set_ylabel("건수")
axes[1].set_title("모델별 2016 홀드아웃 RMSE / MAE")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=150)
plt.close()
print(f"saved: {FIG_OUT}")
