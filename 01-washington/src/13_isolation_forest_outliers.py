"""
Isolation Forest로 이상치(이례적인 날)를 탐지하고, 중앙값 치환(median ablation) 방식으로
각 이상치의 직접 원인 변수를 특정한다.

입력: data/day_processed.csv
출력:
  - tables/isolation_forest_outliers.csv (이상치 37일 + 원인 변수 + 이상 점수)
  - docs/outlier_dashboard.html (시각화 대시보드, 별도 빌드 스크립트로 생성됨)
"""
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["temp", "atemp", "hum", "windspeed", "weathersit", "cnt"]
LABELS_KO = {
    "temp": "기온", "atemp": "체감온도", "hum": "습도",
    "windspeed": "풍속", "weathersit": "날씨등급", "cnt": "대여수",
}


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    iso.fit(X)

    df = df.copy()
    df["anomaly_score"] = iso.decision_function(X)
    df["is_outlier"] = iso.predict(X) == -1
    return df, iso, X


def attribute_cause(df: pd.DataFrame, iso: IsolationForest, X: pd.DataFrame) -> pd.DataFrame:
    """각 이상치에서 어떤 변수가 원인인지, 해당 변수를 중앙값으로 바꿨을 때
    이상 점수가 얼마나 정상 쪽으로 회복되는지로 판정한다."""
    median = X.median()
    outliers = df[df["is_outlier"]].sort_values("anomaly_score").copy()

    rows = []
    for idx in outliers.index:
        orig_row = X.loc[idx].copy()
        orig_score = iso.decision_function(orig_row.to_frame().T)[0]

        deltas = {}
        for f in FEATURES:
            modified = orig_row.copy()
            modified[f] = median[f]
            new_score = iso.decision_function(modified.to_frame().T)[0]
            deltas[f] = new_score - orig_score

        top_feat = max(deltas, key=deltas.get)
        sorted_feats = sorted(deltas.items(), key=lambda x: -x[1])

        rows.append({
            "dteday": df.loc[idx, "dteday"],
            "anomaly_score": round(orig_score, 4),
            "top_cause": LABELS_KO[top_feat],
            "top_cause_value": round(orig_row[top_feat], 3),
            "top_cause_median": round(median[top_feat], 3),
            "2nd_cause": LABELS_KO[sorted_feats[1][0]],
            "weathersit": int(df.loc[idx, "weathersit"]),
            "cnt": int(df.loc[idx, "cnt"]),
        })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("data/day_processed.csv")
    df, iso, X = detect_outliers(df)

    n_outliers = int(df["is_outlier"].sum())
    print(f"전체 {len(df)}건 중 이상치 {n_outliers}건 ({n_outliers/len(df)*100:.1f}%)")

    result = attribute_cause(df, iso, X)
    print(result["top_cause"].value_counts())

    result.to_csv("tables/isolation_forest_outliers.csv", index=False, encoding="utf-8-sig")
    print("saved: tables/isolation_forest_outliers.csv")


if __name__ == "__main__":
    main()
