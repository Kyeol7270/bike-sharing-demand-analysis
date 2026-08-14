"""hum(습도)=0 인 행을 결측치로 간주하고 시간축 기준 선형보간.
원본 파일(data/day.csv)은 수정하지 않고, 보간 결과는 별도 파일로 저장.
"""
import pandas as pd

SRC = "data/day.csv"
DST = "data/day_processed.csv"

df = pd.read_csv(SRC)
df = df.sort_values("instant").reset_index(drop=True)

before = df.loc[df["hum"] == 0, ["instant", "dteday", "hum"]]
print("=== 보간 전 (hum=0) ===")
print(before.to_string(index=False))

df.loc[df["hum"] == 0, "hum"] = pd.NA
df["hum"] = df["hum"].astype(float).interpolate(method="linear")

after = df.loc[df["instant"].isin(before["instant"]), ["instant", "dteday", "hum"]]
print()
print("=== 보간 후 ===")
print(after.to_string(index=False))

df.to_csv(DST, index=False)
print()
print(f"saved: {DST}")
