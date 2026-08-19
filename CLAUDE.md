# 프로젝트 개요

자전거 대여 데이터셋(UCI Bike Sharing Dataset, 일별)을 활용한 데이터 분석 · 시각화 · 회귀분석 프로젝트.

## 폴더 구조

```
19th/
├── inputs/     # 입력 데이터
│   ├── day.csv            # 원본 (절대 직접 수정하지 않음)
│   └── day_processed.csv  # 가공본: hum=0(instant 69) 결측 처리 후 시간축 선형보간
├── scripts/    # 분석 코드 (.py)
├── outputs/    # 분석 결과물
│   ├── figures/   # 시각화 이미지 (png)
│   ├── tables/    # 결과 표, 통계 요약, 모델 성능 지표 (csv)
│   └── models/    # 학습된 모델 파일 (joblib)
├── docs/       # 문서
│   └── worklog.md   # 작업 이력 로그 (최신 항목이 맨 위, 상단 10줄만으로 직전/다음 작업 파악 가능하게 작성)
└── CLAUDE.md   # 이 파일. 프로젝트 컨텍스트 요약
```

- 분석 스크립트는 `scripts/`에 작성. 데이터를 읽을 때는 `inputs/`, 결과를 저장할 때는 `outputs/` 하위 폴더를 기준 경로로 사용.
- git 저장소는 아직 초기화하지 않음 (`git init`은 사용자 요청 시 진행 — 리마인드 예정 상태).

## 데이터셋: inputs/day.csv / inputs/day_processed.csv

UCI Bike Sharing Dataset (일별 집계, 731행). `day_processed.csv`는 `day.csv`에서 `hum=0`이었던 1건(instant 69, 2011-03-10)만 결측 처리 후 보간한 버전이며, 그 외 컬럼은 원본과 동일. 이후 분석은 `day_processed.csv` 사용.

| 컬럼 | 설명 |
|---|---|
| instant | 레코드 인덱스 |
| dteday | 날짜 |
| season | 계절 (1:봄, 2:여름, 3:가을, 4:겨울) |
| yr | 연도 (0:2011, 1:2012) |
| mnth | 월 (1~12) |
| holiday | 공휴일 여부 |
| weekday | 요일 (0~6) |
| workingday | 평일 여부 |
| weathersit | 날씨 상태 (1:맑음 ~ 4:악천후) |
| temp | 정규화된 기온 |
| atemp | 정규화된 체감온도 |
| hum | 정규화된 습도 |
| windspeed | 정규화된 풍속 |
| casual | 비회원 대여 건수 |
| registered | 회원 대여 건수 |
| cnt | 총 대여 건수 (casual + registered, 타겟 변수 후보) |

## 진행 현황

1. **EDA**: 완료 — 결측치 0건, IQR 기준 이상치 확인 (hum=0 1건→보간, windspeed 13건, casual 44건은 유지)
2. **시각화**: 완료 — 온도-대여건수 산점도, 날씨등급별 박스플롯, 계절/요일별 트렌드, 계절x요일 히트맵 (`outputs/figures/`)
3. **회귀분석**: 1차 비교 → 과적합/CV 검증 → 하이퍼파라미터 튜닝 → 최종 비교 시각화까지 완료. 타겟 `cnt`, 설명변수 11개(season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed)

   | 모델 | CV R² (5-fold) | CV RMSE | CV MAE | 비고 |
   |---|---|---|---|---|
   | **XGBoost (튜닝)** | **0.893** | 621.66 | 439.36 | **최종 채택** — 튜닝으로 과적합 격차 0.116→0.082, 성능도 함께 개선 |
   | RandomForest (default) | 0.875 | 673.24 | 457.42 | 튜닝해도 개선 없어 default 유지 |
   | RandomForest (튜닝) | 0.874 | 676.85 | 465.15 | 참고용, 미채택 (default와 사실상 동일) |
   | XGBoost (default) | 0.872 | 681.54 | 474.62 | 튜닝 전 baseline |
   | LinearRegression (원-핫) | 0.821 | 805.42 | 586.77 | 과적합 없음(gap 0.007), 해석 용도로 유효 |
   | LinearRegression (순서형) | 0.783 | 887.61 | 656.23 | 최초 baseline |

   **최종 결론: XGBoost(튜닝)이 3개 지표(R²/RMSE/MAE) 모두에서 1위** — `outputs/figures/model_performance_comparison.png`로 시각 확인.
   튜닝 방법: RandomizedSearchCV, 60개 조합 × 5-fold CV = 300회 학습. (Windows에서 사용자 경로에 한글이 포함되어 있어 `n_jobs=-1` 사용 시 joblib 멀티프로세싱이 `UnicodeEncodeError`로 실패 — 항상 `n_jobs=1`로 실행할 것)

4. **Feature Importance 분석**: 완료 (`scripts/feature_importance.py` → `outputs/figures/feature_importance_xgboost.png`) — 최종 채택 모델(XGBoost 튜닝) 기준, gain importance + permutation importance 교차 확인

   | 순위 | 변수 | Permutation Importance |
   |---|---|---|
   | 1 | **yr (연도)** | 0.507 |
   | 2 | **temp (기온)** | 0.345 |
   | 3 | hum (습도) | 0.103 |
   | 4 | season (계절) | 0.066 |
   | 5 이하 | windspeed, mnth, weathersit, atemp, workingday, weekday, holiday | 0.003~0.035 |

   `yr`이 압도적 1위 — 날씨보다 "연도별 서비스 성장 추세"(2011→2012 이용자 기반 확대)가 예측에 더 크게 기여. 날씨 변수 중에서는 `temp`가 가장 중요하고 `hum`이 의외로 `season`보다 중요하게 나타남(계절 정보가 mnth/weathersit/temp와 겹쳐 개별 기여도가 분산된 것으로 해석). `weekday`/`holiday`/`workingday`는 개별 기여도는 낮음(계절과의 조합 효과는 히트맵에서 별도 확인됨).

   다음 후보: temp-atemp 다중공선성(VIF) 점검, 예측 vs 실제 잔차 플롯.

## 모델링 규칙

- 타겟은 `cnt`. **`casual`, `registered`는 설명변수에서 반드시 제외** — `cnt = casual + registered`이므로 포함 시 데이터 누수(R²=1.0으로 왜곡) 발생.
- train/test split은 `test_size=0.2, random_state=42`로 통일 (모델 간 성능 비교 시 동일 split 사용).
- 트리 기반 모델(RandomForest, XGBoost)은 원-핫 인코딩 없이 순서형 인코딩 그대로 사용.
- 하이퍼파라미터 탐색은 RandomizedSearchCV(5-fold CV) 사용, `n_iter=60`. **`n_jobs`는 반드시 `1`로 지정** — 이 프로젝트 경로에 한글(사용자명)이 포함돼 있어 병렬 처리(`n_jobs=-1`) 시 joblib이 `UnicodeEncodeError`로 실패함.
- **pandas DataFrame을 만들 때 정렬된 Series와 정렬 안 된 numpy 배열을 섞지 말 것** — `pd.DataFrame({"a": some_sorted_series, "b": some_raw_array})`처럼 만들면 배열이 Series의 정렬된 인덱스 순서에 맞춰 잘못 배치되는 버그가 발생함(실제로 feature importance 분석에서 발생, 차트 라벨-값 불일치로 발견). 여러 소스를 합칠 때는 전부 동일한 인덱스(`pd.Series(array, index=...)`)를 명시할 것.

## 작업 규칙

- 새로운 분석/작업을 시작하거나 마칠 때 `docs/worklog.md`에 기록한다. **최신 항목을 파일 맨 위에 추가**하고, 상단 10줄만 봐도 직전 작업과 다음 작업을 알 수 있도록 요약한다.
- 원본 데이터(`inputs/day.csv`)는 수정하지 않고, 가공 데이터가 필요하면 별도 파일로 `inputs/` 또는 `outputs/`에 저장한다.
