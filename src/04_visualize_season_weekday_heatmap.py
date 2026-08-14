"""계절 x 요일 조합별 평균 대여 건수(cnt) 히트맵.
원본 데이터(data/day.csv)는 읽기 전용으로만 사용, 수정하지 않음.
결과는 figures/ 에 저장.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#525142" if False else "#52514e"
SURFACE = "#fcfcfb"

# sequential 블루 램프 (100~700 steps)
SEQ_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
             "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
             "#184f95", "#104281", "#0d366b"]
CMAP = mcolors.LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

DATA_PATH = "data/day.csv"
OUT_DIR = "figures"

df = pd.read_csv(DATA_PATH)

season_labels = {1: "봄", 2: "여름", 3: "가을", 4: "겨울"}
weekday_labels = {0: "일", 1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토"}

pivot = df.pivot_table(index="season", columns="weekday", values="cnt", aggfunc="mean")
pivot = pivot.reindex(index=[1, 2, 3, 4], columns=[0, 1, 2, 3, 4, 5, 6])

fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor=SURFACE)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

im = ax.imshow(pivot.values, cmap=CMAP, aspect="auto")

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([weekday_labels[d] for d in pivot.columns], fontsize=10, color=INK_SECONDARY)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([season_labels[s] for s in pivot.index], fontsize=10, color=INK_SECONDARY)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# 셀 사이 여백(surface gap) 표현 + 값 라벨
vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
mid = (vmin + vmax) / 2
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor=SURFACE, linewidth=2))
        text_color = "#ffffff" if v > mid else INK_PRIMARY
        ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=9, color=text_color)

ax.set_title("계절 x 요일별 평균 대여 건수(cnt)", fontsize=12, color=INK_PRIMARY, pad=14)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.outline.set_visible(False)
cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
cbar.set_label("평균 cnt", fontsize=9, color=INK_SECONDARY)

fig.tight_layout()
fig.savefig(f"{OUT_DIR}/season_weekday_heatmap.png", facecolor=SURFACE)
plt.close(fig)

print(f"saved: {OUT_DIR}/season_weekday_heatmap.png")
