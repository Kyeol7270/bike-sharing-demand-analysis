"""EDA 시각화: 온도-대여건수 관계 / 날씨등급별 비교 / 계절·요일별 경향.
원본 데이터(data/day.csv)는 읽기 전용으로만 사용, 수정하지 않음.
결과는 figures/ 에 저장.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# --- 팔레트 (참고: dataviz skill 기본 팔레트) ---
BLUE = "#2a78d6"      # sequential / categorical slot1
ORANGE = "#eb6834"    # categorical slot2
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SEQ_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]  # ordinal steps: 250/300/450/550 근사

DATA_PATH = "data/day.csv"
OUT_DIR = "figures"

df = pd.read_csv(DATA_PATH)


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ============================================================
# 1. 온도 vs 대여 건수 (산점도 + 추세선)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor=SURFACE)
style_ax(ax)

x = df["temp"]
y = df["cnt"]
ax.scatter(x, y, s=16, color=BLUE, alpha=0.55, edgecolors="none", zorder=2)

# 선형 추세선
coef = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, np.polyval(coef, xs), color=INK_PRIMARY, linewidth=1.5, zorder=3)

corr = df["temp"].corr(df["cnt"])
ax.set_title("정규화 기온(temp)과 총 대여 건수(cnt)의 관계", fontsize=12, color=INK_PRIMARY, pad=12)
ax.set_xlabel("temp (정규화 기온)", fontsize=10, color=INK_SECONDARY)
ax.set_ylabel("cnt (총 대여 건수)", fontsize=10, color=INK_SECONDARY)
ax.text(0.03, 0.95, f"상관계수 r = {corr:.2f}", transform=ax.transAxes,
        fontsize=10, color=INK_SECONDARY, va="top")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/temp_vs_cnt_scatter.png", facecolor=SURFACE)
plt.close(fig)


# ============================================================
# 2. 날씨 등급별(weathersit) 대여 건수 비교 (박스플롯, 순서형 → 단일색 그라데이션)
# ============================================================
weather_labels = {1: "1. 맑음", 2: "2. 안개/흐림", 3: "3. 약한 눈/비"}
present = sorted(df["weathersit"].unique())

fig, ax = plt.subplots(figsize=(7, 5), dpi=150, facecolor=SURFACE)
style_ax(ax)

data_by_group = [df.loc[df["weathersit"] == w, "cnt"].values for w in present]
bp = ax.boxplot(data_by_group, patch_artist=True, widths=0.5,
                 medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
                 whiskerprops=dict(color=INK_MUTED),
                 capprops=dict(color=INK_MUTED),
                 flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_MUTED,
                                  markeredgecolor="none", alpha=0.6))

ramp = [SEQ_RAMP[i] for i in np.linspace(1, len(SEQ_RAMP) - 1, len(present)).astype(int)]
for patch, color in zip(bp["boxes"], ramp):
    patch.set_facecolor(color)
    patch.set_edgecolor(INK_SECONDARY)
    patch.set_linewidth(0.8)

ax.set_xticks(range(1, len(present) + 1))
ax.set_xticklabels([weather_labels.get(w, str(w)) for w in present], fontsize=10, color=INK_SECONDARY)
ax.set_title("날씨 등급(weathersit)별 대여 건수(cnt) 분포", fontsize=12, color=INK_PRIMARY, pad=12)
ax.set_ylabel("cnt (총 대여 건수)", fontsize=10, color=INK_SECONDARY)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/weathersit_cnt_boxplot.png", facecolor=SURFACE)
plt.close(fig)


# ============================================================
# 3. 계절별 / 요일별 경향
# ============================================================
season_labels = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
weekday_labels = {0: "일", 1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, facecolor=SURFACE)

# 3-1. 계절별 평균 대여 건수 (순서형 → 단일 색 그라데이션)
ax = axes[0]
style_ax(ax)
season_mean = df.groupby("season")["cnt"].mean().reindex([1, 2, 3, 4])
ramp4 = [SEQ_RAMP[i] for i in np.linspace(1, len(SEQ_RAMP) - 1, 4).astype(int)]
bars = ax.bar([season_labels[s] for s in season_mean.index], season_mean.values,
              color=ramp4, edgecolor=INK_SECONDARY, linewidth=0.6, width=0.6, zorder=2)
for b, v in zip(bars, season_mean.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,.0f}", ha="center",
            fontsize=9, color=INK_SECONDARY)
ax.set_title("계절별 평균 대여 건수", fontsize=12, color=INK_PRIMARY, pad=12)
ax.set_ylabel("평균 cnt", fontsize=10, color=INK_SECONDARY)

# 3-2. 요일별 평균 대여 건수 (회원 vs 비회원, 카테고리컬 2계열)
ax = axes[1]
style_ax(ax)
wk = df.groupby("weekday")[["registered", "casual"]].mean().reindex([0, 1, 2, 3, 4, 5, 6])
xpos = np.arange(7)
width = 0.38
ax.bar(xpos - width / 2, wk["registered"], width=width, color=BLUE, label="registered (회원)", zorder=2)
ax.bar(xpos + width / 2, wk["casual"], width=width, color=ORANGE, label="casual (비회원)", zorder=2)
ax.set_xticks(xpos)
ax.set_xticklabels([weekday_labels[d] for d in wk.index], fontsize=10, color=INK_SECONDARY)
ax.set_title("요일별 평균 대여 건수 (회원/비회원)", fontsize=12, color=INK_PRIMARY, pad=12)
ax.set_ylabel("평균 건수", fontsize=10, color=INK_SECONDARY)
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/season_weekday_trend.png", facecolor=SURFACE)
plt.close(fig)

print("saved:")
print(f"  {OUT_DIR}/temp_vs_cnt_scatter.png")
print(f"  {OUT_DIR}/weathersit_cnt_boxplot.png")
print(f"  {OUT_DIR}/season_weekday_trend.png")
