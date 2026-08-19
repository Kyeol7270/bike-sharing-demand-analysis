"""
시간별(london_merged.csv) -> 일별 변환.

계획 문서: 02-london/src/day_conversion_plan.md

원본 파일(london_merged.csv)은 절대 수정하지 않고 읽기 전용으로만 사용한다.
"""

import pandas as pd

SRC = "data/london_merged.csv"
OUT = "data/day.csv"

MEAN_COLS = ["t1", "t2", "hum", "wind_speed"]
SUM_COLS = ["cnt"]
CONST_COLS = ["season", "is_holiday", "is_weekend"]


def worst_code_run_length(hours: pd.Series, codes: pd.Series, target_code: float) -> int:
    """target_code가 연속된 시간(hour) 값으로 몇 시간 이어지는지 최댓값을 구한다.
    시간이 결측으로 끊기면(연속 hour 값이 아니면) streak이 끊긴 것으로 본다."""
    mask = (codes == target_code).to_numpy()
    hrs = hours.to_numpy()
    best = cur = 0
    for i in range(len(hrs)):
        if not mask[i]:
            cur = 0
            continue
        if i > 0 and mask[i - 1] and hrs[i] == hrs[i - 1] + 1:
            cur += 1
        else:
            cur = 1
        best = max(best, cur)
    return best


def aggregate_weather_code(group: pd.DataFrame) -> float:
    g = group.sort_values("hour")
    worst = g["weather_code"].max()

    if worst_code_run_length(g["hour"], g["weather_code"], worst) >= 4:
        return worst

    counts = g["weather_code"].value_counts()
    max_freq = counts.max()
    tied = counts[counts == max_freq].index
    return max(tied)


def main():
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # 사전 검증: 날짜 속성형 컬럼이 하루 내내 동일한지
    for c in CONST_COLS:
        bad = df.groupby("date")[c].nunique()
        assert (bad == 1).all(), f"{c}가 하루 내에 동일하지 않은 날짜 존재"

    daily = df.groupby("date").agg(
        **{c: (c, "mean") for c in MEAN_COLS},
        **{c: (c, "sum") for c in SUM_COLS},
        **{c: (c, "first") for c in CONST_COLS},
        n_hours_observed=("cnt", "count"),
    )
    for c in MEAN_COLS:
        daily[c] = daily[c].round(1)

    weather = df.groupby("date").apply(aggregate_weather_code, include_groups=False)
    daily["weather_code"] = weather

    daily = daily.reset_index()

    date_dt = pd.to_datetime(daily["date"])
    daily.insert(0, "instant", range(1, len(daily) + 1))
    daily["year"] = date_dt.dt.year
    daily["month"] = date_dt.dt.month
    daily["day"] = date_dt.dt.day
    daily["weekday"] = date_dt.dt.day_name()
    daily["working"] = ((daily["is_holiday"] == 0) & (daily["is_weekend"] == 0)).astype(int)

    cols = ["instant", "year", "month", "day", "weekday", "season", "is_holiday",
            "is_weekend", "working", "weather_code", "t1", "t2", "hum", "wind_speed",
            "cnt", "n_hours_observed"]
    daily = daily[cols]

    daily.to_csv(OUT, index=False)
    print(f"{len(daily)}행 저장 -> {OUT}")
    print(f"결측 시간이 있는 날: {(daily['n_hours_observed'] < 24).sum()}건")


if __name__ == "__main__":
    main()
