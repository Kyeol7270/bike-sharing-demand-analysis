# 런던 자전거 대여 분석 통합 계획

> 목표: **2015년 데이터로 회귀모델을 학습해 2016년 대여 건수(`cnt`)를 예측**하는 최적 모델을 찾는다.
> `01-washington/`에서 완료한 워크플로우(전처리 → EDA → 회귀분석 → 이상치 탐지 → 최종 리포트)를
> 참고해 런던 데이터에 맞게 재구성한 계획이며, 아직 실행 전(계획 단계)이다.

- train: `data/day_2015.csv` (362행, 2015-01-04~2015-12-31)
- test: `data/day_2016.csv` (365행, 2016-01-01~2016-12-31)

## 0. 워싱턴 워크플로우 요약 (참고 기준)

`01-washington/src/`의 14개 스크립트를 단계별로 정리하면:

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 전처리 | 01_data_overview, 02_clean_missing_values | shape/dtype/결측치 확인, 물리적으로 불가능한 값(hum=0) 발견 후 선형보간 |
| EDA/시각화 | 03_visualize_eda, 04_visualize_season_weekday_heatmap | 온도-대여량 산점도, 날씨등급 박스플롯, 계절×요일 히트맵 |
| 회귀분석 | 05_baseline_linear_regression ~ 11_final_model_comparison | baseline → 데이터 누수 체크(casual/registered 제외) → 4개 모델 비교(LinearRegression 순서형/원핫, RandomForest, XGBoost) → 과적합/5-fold CV 검증 → RandomizedSearchCV 튜닝(n_iter=60, **n_jobs=1** 고정) → CV+홀드아웃 종합 비교 |
| 변수 해석 | 12_feature_importance | gain importance + permutation importance 교차 확인 |
| 이상치 탐지 | 13_isolation_forest_outliers | Isolation Forest(contamination=5%) + 중앙값 치환(median ablation)으로 원인 변수 특정 + 실제 기상 이벤트와 대조 |
| 오차 분석 | 14_prediction_error_analysis | 예측 오차 상위 10일과 이상치 목록 교집합 확인 |
| 리포트 | README.md, docs/REPORT.md | 결과·판단 근거·한계를 정리, 수치는 항상 원본 재계산(하드코딩 금지) |

이 계획은 이 순서를 뼈대로 삼되, 아래 2절에서 정리한 **런던 데이터의 구조적 차이** 때문에 그대로
복붙할 수 없는 지점들을 미리 표시해둔다.

## 1. 이번 분석이 워싱턴과 근본적으로 다른 점

**train/test가 무작위 분할이 아니라 연도 단위 시간 분할(temporal holdout)이다.** 워싱턴은
`train_test_split(test_size=0.2, random_state=42)`로 731일 전체를 무작위로 섞어 나눴고, README
"한계" 절에 "일별 시계열 자기상관을 고려하면 성능이 낙관적으로 부풀려졌을 가능성이 있다"는 미해결
과제를 남겼다(TimeSeriesSplit 검증이 "다음 단계"로만 남고 실행되지 않음). 이번 런던 분석은 **2015년
전체로 학습 → 2016년 전체를 예측**하는 구조라 그 한계를 처음부터 피해가는 더 엄격한 검증이다. 그만큼
아래 항목들도 "무작위 분할이면 문제없었을" 것들이 실제 문제가 된다.

## 2. 컬럼별 처리 계획 (확정)

`day_2015.csv`/`day_2016.csv` 공통 컬럼: `instant, year, month, day, weekday, season, is_holiday,
is_weekend, working, weather_code, t1, t2, hum, wind_speed, cnt, n_hours_observed`

| 컬럼 | 확정된 처리 | 근거 |
|---|---|---|
| `instant` | 제외 (인덱스일 뿐, 워싱턴도 제외) | 없음 |
| **`year`** | **완전 제외** | train은 2015 하나, test는 2016 하나뿐이라 **각 파일 내에서 상수** — 모델이 학습할 정보가 없고(트리 모델은 상수 컬럼으로 분기 자체가 불가능, 선형회귀는 절편과 완전 공선이라 계수가 사실상 무의미), 포함해도 2016 예측 시 훈련 때 못 본 값(2016)으로 외삽하는 꼴이 됨. 워싱턴에서 `yr`이 중요도 1위(0.507)였던 것과 정반대로, **여기서는 반드시 빼야 하는 컬럼** |
| `month`, `day` | 포함 (season과 함께 계절성 포착) | `day`(1~31, 월중 날짜)는 예측력이 거의 없을 가능성이 높음 — feature importance에서 낮게 나오면 제거 검토 |
| **`weekday`** | **선형회귀: 원-핫 인코딩 / 트리 모델: 달력 순서(월=0~일=6) 정수 인코딩** | 선형회귀에 순서형을 쓰면 "월→화→…→일 갈수록 대여량이 일정하게 증감한다"는 근거 없는 가정을 강제하게 돼 원-핫이 맞음. 트리는 "임의" 순서가 아니라 실제 달력 순서를 쓰면 평일/주말 묶음을 한 번의 분기로 나눌 수 있어(`is_weekend`가 이미 담은 정보와도 상충하지 않음) 순서형이 더 효율적 |
| `season` | 원본 값(0~3) 그대로 사용, 선형회귀는 원-핫도 비교 | 워싱턴과 값 범위가 달라(1~4 vs 0~3) 계수 해석 시 착오 주의 |
| `is_holiday`, `is_weekend` | **둘 다 사용** | `working`은 이 둘의 결정론적 함수(`(is_holiday==0)&(is_weekend==0)`)임을 실측 확인(362행 전부 일치) — `working`만 쓰면 "휴일이라 쉬는 날"과 "주말이라 쉬는 날"을 구분 못 하고 뭉개버림(워싱턴 별도 분석에서 회원/비회원이 휴일에 정반대로 움직인 전례처럼, 두 효과가 다를 수 있음). 정보 손실 없이 다중공선성만 없애려면 이 조합이 적합 |
| `working` | **제외** | 위 사유로 `is_holiday`+`is_weekend`가 정보량이 더 많고 겹치는 컬럼이라 제거 |
| `weather_code` | 트리 모델은 원본 코드 그대로(순서형만), 선형회귀는 원-핫 vs 순서형 두 버전 모두 비교 | 워싱턴에서 `weathersit` 원-핫이 선형회귀 CV R²를 0.783→0.821로 유의하게 올렸던 전례가 있어 같은 실험을 반복(코드 체계가 달라 결과가 같을지는 실측 필요). 트리는 인코딩에 덜 민감해 순서형 하나로 충분 |
| `t1`, `t2` | 둘 다 포함 후 상관관계/VIF 확인 | 체감온도(t2)와 실제기온(t1)은 상관이 매우 높을 가능성 — 워싱턴 temp/atemp도 같은 이슈가 있었으나 명시적으로 다루지 않았음(README "다음 단계"에 VIF 확인이 미착수로 남아있음). 이번엔 먼저 확인 |
| `hum`, `wind_speed` | 포함 | 없음 |
| `n_hours_observed` | **모델 피처로 넣지 않되, 날짜(행) 자체는 삭제하지 않음** — 결측 많은 날(34일, 최저 9시간) 목록을 부록으로 남겨 진단용으로만 활용 | train이 362일뿐이라 저신뢰 날짜를 통째로 빼면 이미 작은 데이터가 더 줄어듦. 워싱턴도 결측 시간이 있는 날을 빼지 않고 존재하는 값만으로 집계했던 전례를 따름. 필요하면 추후 "저신뢰 날짜 제외 시 성능 변화"를 별도 민감도 분석으로 확인 |
| `cnt` | 타깃 | `casual`/`registered`가 원본에 없어 워싱턴의 "데이터 누수 체크"(06번 스크립트) 자체가 필요 없음 — 대신 위 `working` 제외 판단이 그 자리를 대체 |

## 3. 단계별 실행 계획

### 3-1. 전처리 상태 확인 (이미 완료, 재확인만)
- `day_2015.csv`/`day_2016.csv` 결측치 0건, `n_hours_observed`로 저신뢰 날짜(34일) 표시 확인.
- train(2015)과 test(2016) 각각에서 `season`/`is_holiday`/`is_weekend` 등 날짜 속성형 분포가
  한쪽으로 심하게 쏠려있지 않은지 확인(train에만 있는 계절/공휴일 조합이 있으면 일반화가 어려움).

### 3-2. EDA
- 기술통계, `cnt` 분포(첨부: train 기준 평균 26,903 / 표준편차 8,390 / 최대 72,504 — 왜도 확인 필요).
- `t1` vs `cnt` 산점도, `weather_code`별 `cnt` 박스플롯, `season`×`weekday` 히트맵 (워싱턴
  04번 스크립트와 동일한 형식으로 제작, 회원/비회원 구분은 없으므로 총 `cnt` 하나만 봄).
- train(2015)과 test(2016)의 `cnt` 분포·계절 패턴을 나란히 비교해 **연도 간 분포 이동(distribution
  shift)** 이 있는지 확인 — 있다면 모델이 2016에서 성능이 떨어질 수 있는 원인으로 미리 인지.

### 3-3. 베이스라인 및 다중공선성 점검
- `is_holiday` + `is_weekend`(`working` 제외)로 선형회귀 baseline 학습 (train=2015).
- VIF로 `t1`/`t2` 등 연속형 변수 간 다중공선성 사전 확인 (워싱턴에서 미착수였던 부분을 여기서 먼저 함).

### 3-4. 모델 비교
- LinearRegression(순서형: season/weather_code/weekday 모두 정수, weekday는 달력 순서) /
  LinearRegression(원-핫: season, month, weekday, weather_code) / RandomForest / XGBoost, 4종
  비교 (워싱턴 07번과 동일한 구조). 트리 모델(RandomForest/XGBoost)은 weekday만 달력 순서 정수,
  weather_code/season은 원본 값 그대로 사용(원-핫 비교는 선형회귀에서만 진행).
- **평가 방식**: 워싱턴처럼 `train_test_split`으로 한 파일을 다시 쪼개지 않는다. **학습·튜닝은
  2015 데이터 내부 K-fold CV로만 진행**하고, **2016(day_2016.csv)은 최종 모델 확정 전까지 절대
  들여다보지 않는 진짜 홀드아웃**으로 둔다. 이는 워싱턴이 README에 남겨둔 "무작위 분할이 자기상관
  때문에 성능을 부풀렸을 가능성" 한계를 이번 분석에서 처음부터 피하는 지점이다.

### 3-5. 과적합 검증 및 튜닝
- train(2015) 내부에서 train/validation R² 격차로 과적합 확인 (워싱턴 08번과 동일한 방식).
- RandomizedSearchCV(5-fold CV, n_iter=60)로 RandomForest/XGBoost 튜닝. **`n_jobs=1` 고정** —
  워싱턴과 동일하게 이 저장소 경로에 한글이 포함돼 있어 `n_jobs=-1` 사용 시 joblib
  멀티프로세싱이 `UnicodeEncodeError`로 실패하는 환경 이슈가 그대로 적용됨.

### 3-6. 최종 모델 확정 및 2016 예측
- 튜닝까지 끝낸 각 모델을 **2015 전체로 재학습 → 2016 전체를 한 번만 예측**해서 R²/RMSE/MAE 산출.
- CV 점수(2015 내부)와 홀드아웃 점수(2016)를 나란히 표로 제시하고, 워싱턴 README처럼 "CV 1위 ≠
  홀드아웃 1위"일 가능성을 열어두고 단정하지 않는다.
- 결과표는 하드코딩하지 않고 스크립트가 원본에서 매번 재계산하도록 작성(워싱턴 11번 스크립트가
  외부 리뷰로 하드코딩 문제를 지적받아 재작성했던 사례를 재발 방지 차원에서 처음부터 반영).

### 3-7. 변수 중요도
- 최종 채택 모델 기준 gain importance + permutation importance 교차 확인 (워싱턴 12번과 동일).
- `year`를 애초에 뺐으므로 워싱턴처럼 "연도가 1위"로 나오는 일은 없음 — 대신 `t1`/`weather_code`/
  `season` 중 어떤 게 1위로 나오는지가 이번 분석의 핵심 관전 포인트.

### 3-8. 이상치 탐지
- 2015(train) 기준으로 Isolation Forest(contamination=5%) 이상치 탐지, 중앙값 치환으로 원인 변수
  특정 (워싱턴 13번과 동일한 방법론).
- 상위 이상치 날짜는 실제 런던 기상 이벤트(폭설·폭풍 등)와 대조해 타당성 확인.
- `n_hours_observed`가 낮은 날(결측 많은 날)이 이상치로도 같이 잡히는지 교차 확인 — 두 신호가
  겹치면 "관측 부족이 원인인 이상치"로 별도 표시.

### 3-9. 2016 예측 오차 분석
- 최종 모델로 2016 전체 예측 후 오차 상위 10일 추출 (워싱턴 14번과 동일).
- 2015 이상치 목록과는 대상 기간이 다르므로, **2016 자체에 대해서도 이상치 탐지를 별도로 돌려서**
  오차 상위 날짜와 교집합을 확인해야 함(워싱턴은 같은 데이터셋 안에서 이상치와 오차를 비교했지만,
  런던은 train/test 기간이 분리돼 있어 이 부분을 별도로 설계해야 함).

### 3-10. 최종 리포트
- `README.md`(결과·결론·한계 중심) + `docs/REPORT.md`(상세 근거) 작성, 워싱턴과 동일한 2단 구조.
- 반드시 포함: 모델 성능표(CV vs 2016 홀드아웃), 변수 중요도, 이상치·오차 분석 요약, **`year`를
  뺀 이유와 그로 인한 해석상 제약**(연도별 성장 추세를 애초에 모델링하지 못함), 다음 단계 후보.

## 4. 산출물 구조 (제안)

```
02-london/
├── src/
│   ├── 01_day_conversion.py        # 기완료 (전처리)
│   ├── 02_data_overview.py         # 신규
│   ├── 03_visualize_eda.py
│   ├── 04_multicollinearity_check.py   # is_holiday+is_weekend(working 제외) + t1/t2 VIF
│   ├── 05_baseline_linear_regression.py
│   ├── 06_compare_regression_models.py
│   ├── 07_validate_models_cv_overfitting.py
│   ├── 08_tune_xgboost.py
│   ├── 09_tune_random_forest.py
│   ├── 10_final_model_comparison.py   # 2015 CV vs 2016 홀드아웃
│   ├── 11_feature_importance.py
│   ├── 12_isolation_forest_outliers.py
│   └── 13_prediction_error_analysis.py
├── figures/, tables/, models/          # 워싱턴과 동일한 역할
├── docs/REPORT.md, docs/WORKLOG.md
└── README.md
```

번호·파일명은 실행 시작 전 최종 확정한다(계획 단계이므로 변경 가능).

## 5. 확정된 처리 방침 (요약)

1. `year` 컬럼은 완전 제외한다 — train/test 각각 상수라 정보가 없고, 포함하면 2016 예측 시
   훈련에서 못 본 값으로 외삽하는 꼴이 됨.
2. `working`은 제외하고 `is_holiday` + `is_weekend`를 사용한다 — `working`은 이 둘의 결정론적
   함수라 정보 손실 없이 다중공선성만 없앨 수 있음.
3. `weekday`는 선형회귀엔 원-핫, 트리 모델엔 달력 순서(월=0~일=6) 정수 인코딩을 쓴다.
4. `n_hours_observed`는 모델 피처에서 제외하되 날짜(행)는 삭제하지 않는다 — 저신뢰 날짜 목록만
   부록으로 남겨 진단용으로 활용한다.
5. `weather_code`/`season` 원-핫 vs 순서형 비교는 선형회귀에서만 두 방식 모두 실험하고, 트리
   모델은 순서형만 사용한다.

이 계획서는 실행 전 상태이며, 위 방침을 반영해 `src/02_...`부터 순차적으로 구현할 예정이다.
