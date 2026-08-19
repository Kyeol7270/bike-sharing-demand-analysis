"""EDA 시각화 (train=2015 기준). ANALYSIS_PLAN.md 3-2절.
1) t1 vs cnt 산점도
2) weather_code별 cnt 박스플롯
3) season x weekday 히트맵 (cnt 평균)
4) train vs test cnt 분포 비교 (연도 간 분포 이동 확인)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TRAIN_PATH = "data/day_2015.csv"
TEST_PATH = "data/day_2016.csv"
FIG_DIR = "figures"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 1) t1 vs cnt
plt.figure(figsize=(7, 5))
sns.scatterplot(data=train, x="t1", y="cnt", hue="season", palette="viridis", alpha=0.7)
plt.title("t1(기온) vs cnt (train=2015)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/temp_vs_cnt_scatter.png", dpi=150)
plt.close()

# 2) weather_code boxplot
plt.figure(figsize=(7, 5))
order = sorted(train["weather_code"].unique())
sns.boxplot(data=train, x="weather_code", y="cnt", order=order)
plt.title("weather_code별 cnt 분포 (train=2015)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/weathercode_cnt_boxplot.png", dpi=150)
plt.close()

# 3) season x weekday 히트맵
pivot = train.pivot_table(index="season", columns="weekday", values="cnt", aggfunc="mean")
pivot = pivot[[c for c in WEEKDAY_ORDER if c in pivot.columns]]
plt.figure(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title("season x weekday 평균 cnt (train=2015)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/season_weekday_heatmap.png", dpi=150)
plt.close()

# 4) train vs test cnt 분포 비교
plt.figure(figsize=(7, 5))
sns.kdeplot(train["cnt"], label="train (2015)", fill=True, alpha=0.3)
sns.kdeplot(test["cnt"], label="test (2016)", fill=True, alpha=0.3)
plt.title("cnt 분포 비교: train(2015) vs test(2016)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/train_test_cnt_distribution.png", dpi=150)
plt.close()

print("저장 완료:")
for f in ["temp_vs_cnt_scatter.png", "weathercode_cnt_boxplot.png",
          "season_weekday_heatmap.png", "train_test_cnt_distribution.png"]:
    print(f" - {FIG_DIR}/{f}")
