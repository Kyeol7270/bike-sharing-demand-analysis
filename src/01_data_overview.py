"""데이터 구조 파악: 행/열 개수, 컬럼별 타입, 결측치, 기술통계, IQR 기준 이상치.
데이터: data/day.csv (UCI Bike Sharing Dataset, 일별 집계)
"""
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("data/day.csv")

print("=== 1. 행/열 개수 ===")
print(f"행(rows): {df.shape[0]}")
print(f"열(columns): {df.shape[1]}")

print()
print("=== 2. 컬럼명 / 데이터 타입 ===")
info = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
print(info.to_string(index=False))

print()
print("=== 3. 결측치 개수 ===")
na = df.isna().sum().reset_index()
na.columns = ["column", "missing_count"]
print(na.to_string(index=False))

print()
print("=== 4. 상위 데이터 미리보기 (head 5) ===")
print(df.head())

print()
print("=== 5. 기술통계 (describe) ===")
print(df.describe().to_string())

print()
print("=== 6. IQR 기준 이상치 개수 ===")
cols = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
rows = []
for c in cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[c] < lower) | (df[c] > upper)]
    rows.append({"column": c, "Q1": round(q1, 4), "Q3": round(q3, 4),
                  "lower_bound": round(lower, 4), "upper_bound": round(upper, 4),
                  "outlier_count": len(outliers)})
print(pd.DataFrame(rows).to_string(index=False))

print()
print("=== 7. hum=0 이상치 행 (센서 오류 의심) ===")
print(df[df["hum"] == 0].to_string(index=False))
