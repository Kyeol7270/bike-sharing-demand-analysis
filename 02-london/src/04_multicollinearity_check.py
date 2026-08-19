"""다중공선성 점검. ANALYSIS_PLAN.md 3-3절.
1) working = f(is_holiday, is_weekend) 결정론적 관계 재확인 (working은 피처에서 제외됨을 재검증)
2) 연속형 변수(t1, t2, hum, wind_speed, month, day) VIF 확인
"""
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

TRAIN_PATH = "data/day_2015.csv"

train = pd.read_csv(TRAIN_PATH)

print("=== working = (is_holiday==0)&(is_weekend==0) 재검증 ===")
derived = ((train["is_holiday"] == 0) & (train["is_weekend"] == 0)).astype(int)
print("일치:", (derived == train["working"]).all(), f"({len(train)}행)")
print()

print("=== 연속형 변수 VIF (t1, t2, hum, wind_speed, month, day) ===")
cont_cols = ["t1", "t2", "hum", "wind_speed", "month", "day"]
X = add_constant(train[cont_cols])
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
})
print(vif.to_string(index=False))
print()
print("(통상 VIF>10이면 심각한 다중공선성으로 판단)")
