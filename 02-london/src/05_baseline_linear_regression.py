"""선형회귀 baseline (순서형 인코딩). ANALYSIS_PLAN.md 3-3절.

VIF 점검(04번 스크립트) 결과 t1-t2 VIF가 88~91로 심각한 다중공선성이 확인돼,
선형회귀 계열(이 스크립트 포함 05~06, 09)에서는 t2(체감기온)를 제외하고 t1만 사용한다.
트리 모델(RandomForest/XGBoost)은 다중공선성에 영향을 받지 않으므로 t1/t2 둘 다 사용한다
(ANALYSIS_PLAN.md에 이미 반영됨).

평가는 홀드아웃(day_2016.csv)을 건드리지 않고 train(2015) 내부 5-fold CV로만 진행한다.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

TRAIN_PATH = "data/day_2015.csv"
RANDOM_STATE = 42

WEEKDAY_ORDER = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}

train = pd.read_csv(TRAIN_PATH)
train["weekday_ord"] = train["weekday"].map(WEEKDAY_ORDER)

TARGET = "cnt"
FEATURES = ["month", "day", "weekday_ord", "season", "is_holiday", "is_weekend",
            "weather_code", "t1", "hum", "wind_speed"]

X = train[FEATURES]
y = train[TARGET]

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
model = LinearRegression()

r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

print(f"features ({len(FEATURES)}): {FEATURES}")
print(f"train size: {len(X)} (2015 전체, 5-fold CV)")
print(f"CV R2  = {r2_scores.mean():.4f} +/- {r2_scores.std():.4f}")
print(f"CV MAE = {mae_scores.mean():.2f} +/- {mae_scores.std():.2f}")

model.fit(X, y)
coef_df = pd.DataFrame({"feature": FEATURES, "coef": model.coef_})
coef_df = pd.concat(
    [pd.DataFrame({"feature": ["intercept"], "coef": [model.intercept_]}), coef_df],
    ignore_index=True,
)
print()
print("=== 전체 train(2015)으로 학습한 회귀 계수 ===")
print(coef_df.to_string(index=False))
