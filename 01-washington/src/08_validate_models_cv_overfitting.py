"""4개 회귀모델의 과적합(train vs test) 및 교차검증(K-Fold) 확인.
데이터: data/day_processed.csv, 타겟: cnt (casual/registered 제외)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = "data/day_processed.csv"
OUT_TABLE = "tables/model_validation.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

df = pd.read_csv(DATA_PATH)

TARGET = "cnt"
BASE_FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "workingday",
                  "weathersit", "temp", "atemp", "hum", "windspeed"]
CAT_FOR_ONEHOT = ["season", "mnth", "weekday", "weathersit"]

df_onehot = pd.get_dummies(df, columns=CAT_FOR_ONEHOT, drop_first=True)
onehot_features = [c for c in df_onehot.columns
                    if c not in ["instant", "dteday", "casual", "registered", TARGET]]

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
y_train, y_test = train_df[TARGET], test_df[TARGET]

MODELS = {
    "LinearRegression_ordinal": (LinearRegression(), BASE_FEATURES, df[BASE_FEATURES]),
    "LinearRegression_onehot": (LinearRegression(), onehot_features, df_onehot[onehot_features]),
    "RandomForest": (RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
                      BASE_FEATURES, df[BASE_FEATURES]),
    "XGBoost": (XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
                BASE_FEATURES, df[BASE_FEATURES]),
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results = []

for name, (model, feat, X_full) in MODELS.items():
    X_train = X_full.loc[train_df.index]
    X_test = X_full.loc[test_df.index]

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    gap = train_r2 - test_r2

    y_full = df[TARGET]
    cv_r2 = cross_val_score(model.__class__(**model.get_params()), X_full, y_full,
                             cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(model.__class__(**model.get_params()), X_full, y_full,
                                cv=kf, scoring="neg_root_mean_squared_error")
    cv_mae = -cross_val_score(model.__class__(**model.get_params()), X_full, y_full,
                               cv=kf, scoring="neg_mean_absolute_error")

    results.append({
        "model": name,
        "train_R2": train_r2, "test_R2": test_r2, "overfit_gap(R2)": gap,
        "train_RMSE": train_rmse, "test_RMSE": test_rmse,
        "cv_R2_mean": cv_r2.mean(), "cv_R2_std": cv_r2.std(),
        "cv_RMSE_mean": cv_rmse.mean(), "cv_RMSE_std": cv_rmse.std(),
        "cv_MAE_mean": cv_mae.mean(), "cv_MAE_std": cv_mae.std(),
    })

    print(f"[{name}]")
    print(f"  train R2={train_r2:.4f}  test R2={test_r2:.4f}  gap={gap:.4f}")
    print(f"  train RMSE={train_rmse:.2f}  test RMSE={test_rmse:.2f}")
    print(f"  5-fold CV  R2 = {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}  (per-fold: {np.round(cv_r2,3)})")
    print(f"  5-fold CV  RMSE = {cv_rmse.mean():.2f} +/- {cv_rmse.std():.2f}")
    print(f"  5-fold CV  MAE = {cv_mae.mean():.2f} +/- {cv_mae.std():.2f}")
    print()

results_df = pd.DataFrame(results).sort_values("cv_R2_mean", ascending=False)
results_df.to_csv(OUT_TABLE, index=False)
print(results_df.to_string(index=False))
print()
print(f"saved: {OUT_TABLE}")
