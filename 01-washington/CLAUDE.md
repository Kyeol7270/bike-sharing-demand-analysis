# 프로젝트 개요

자전거 대여 데이터셋(UCI Bike Sharing Dataset, 일별)을 활용한 데이터 분석 · 시각화 · 회귀분석 프로젝트.
저장소 루트(`20th/`)는 도시별 분석을 나란히 담는 컨테이너이며, 이 폴더(`01-washington/`)는
그중 워싱턴 D.C. 데이터 분석 전체를 담고 있다. 같은 루트의 `02-london/`은 런던 자전거 대여
데이터셋(시간별)을 2015년→2016년 예측 회귀분석까지 전 과정 완료한 상태다(전처리→EDA→모델비교·
튜닝→변수중요도→이상치탐지→예측오차분석→리포트). 저장소 전체를 소개하는 포트폴리오 진입점은
루트의 `../README.md`이며, 런던 프로젝트 상세는 `02-london/README.md`, `02-london/CLAUDE.md`
참고.

## 폴더 구조

```
20th/
├── 01-washington/   # 이 프로젝트 — 워싱턴 D.C. 분석 (아래 상세)
└── 02-london/       # 런던 데이터셋 — 2015→2016 예측 회귀분석까지 전 과정 완료

01-washington/
├── data/       # 입력 데이터
│   ├── day.csv            # 원본 (절대 직접 수정하지 않음)
│   └── day_processed.csv  # 가공본: hum=0(instant 69, 2011-03-10) 결측 처리 후 선형보간
├── src/        # 분석 코드 (.py), 01~14 번호 순서대로 실행
├── figures/    # 시각화 이미지 (png)
├── tables/     # 결과 표, 통계 요약, 모델 성능 지표 (csv)
├── models/     # 학습된 모델 파일 (joblib)
├── docs/
│   ├── REPORT.md              # 상세 분석 리포트
│   ├── WORKLOG.md             # 작업 이력 로그
│   └── outlier_dashboard.html # 이상치 원인 분석 인터랙티브 대시보드
├── README.md          # 프로젝트 요약 (결과·결론·한계까지 포함, 가장 먼저 볼 문서)
├── requirements.txt
└── CLAUDE.md   # 이 파일. 프로젝트 컨텍스트 요약
```

- 분석 스크립트는 `src/`에 작성. 데이터를 읽을 때는 `data/`, 결과를 저장할 때는 `figures/`·`tables/`·`models/`를 이 폴더(`01-washington/`) 기준 상대경로로 사용.
- git 저장소(`20th/` 루트에 `.git`)로 관리 중. 커밋은 사용자가 명시적으로 요청할 때만 진행.

## 데이터셋: data/day.csv / data/day_processed.csv

UCI Bike Sharing Dataset (일별 집계, 731행, 2011-01-01~2012-12-31). `day_processed.csv`는
`day.csv`에서 `hum=0`이었던 1건(instant 69, 2011-03-10 — weathersit=3인데 습도 0%는 물리적으로
불가능하다고 판단)만 결측 처리 후 보간(→0.712)한 버전이며, 그 외 컬럼은 원본과 동일. 이후 분석은
`day_processed.csv` 사용. `windspeed` 이상치 13건은 레버리지 확인 후 유지(삭제하지 않음).

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
| weathersit | 날씨 상태 (1:맑음 ~ 3:약한 눈/비 — 이 데이터셋에는 4 없음) |
| temp | 정규화된 기온 |
| atemp | 정규화된 체감온도 |
| hum | 정규화된 습도 |
| windspeed | 정규화된 풍속 |
| casual | 비회원 대여 건수 |
| registered | 회원 대여 건수 |
| cnt | 총 대여 건수 (`casual + registered`, 타겟 변수) |

## 진행 현황

1. **EDA**: 완료 — 결측치 0건, IQR 기준 이상치 확인(hum=0 1건→보간, windspeed 13건은 유지)
2. **시각화**: 완료 — 온도-대여건수 산점도, 날씨등급별 박스플롯, 계절/요일별 트렌드, 계절×요일
   히트맵(`figures/`) — 회원은 평일, 비회원은 주말에 몰리는 패턴 확인
3. **회귀분석**: 1차 비교 → 과적합/CV 검증 → 하이퍼파라미터 튜닝 → 최종 비교(CV+홀드아웃)까지 완료.
   타겟 `cnt`, 설명변수 11개(season, yr, mnth, holiday, weekday, workingday, weathersit, temp,
   atemp, hum, windspeed). **`casual`/`registered`는 데이터 누수라 제외** (`src/06_data_leakage_check.py`
   에서 포함 시 R²=1.0으로 왜곡됨을 검증).

   | 모델 | CV R² (5-fold) | CV MAE | 홀드아웃 R² | 홀드아웃 MAE |
   |---|---|---|---|---|
   | **XGBoost (튜닝)** | **0.893 ±0.024** | 439.36 | **0.897** | 435.14 |
   | RandomForest (default, n_est=300) | 0.875 ±0.026 | 457.42 | 0.886 | **426.19 (홀드아웃 MAE 1위)** |
   | RandomForest (튜닝) | 0.874 ±0.025 | 465.15 | 0.882 | 436.56 |
   | XGBoost (default, n_est=300) | 0.872 ±0.024 | 474.62 | 0.884 | 447.39 |
   | LinearRegression (원-핫) | 0.821 ±0.034 | 586.77 | 0.841 | 585.07 |
   | LinearRegression (순서형) | 0.783 ±0.043 | 656.23 | 0.828 | 616.44 |

   (정확한 수치 출처: `tables/model_comparison_final.csv`)

   **"XGBoost가 모든 지표에서 최고"는 아니다.** 상위 트리 모델 4개의 CV R² 차이(0.872~0.893)는
   5-fold 표준편차(±0.024~0.026)보다 작고, 튜닝에 쓰이지 않은 홀드아웃 구간에서는 MAE 1위가
   **RandomForest(default, 426.19)** 다 — 흥미롭게도 튜닝된 RandomForest(436.56)보다도 낮다. 그래서
   CV·홀드아웃 두 프로토콜을 나란히 보고하고 순위를 단정하지 않는다(근거:
   `figures/model_performance_comparison.png`, README "한눈에 보는 결과" 절 참고). **README·리포트에서
   "기준 모델"로 부르는 것은 XGBoost(튜닝)** 이지만, 이는 CV 기준 대표값일 뿐 절대적 1위 선언이
   아님에 유의. 실제로 저장된 모델 파일(`models/`)도 XGBoost(튜닝)과 LinearRegression(baseline)
   뿐이며, RandomForest는 joblib으로 저장돼 있지 않다(스크립트로 재현은 가능).

   튜닝 방법: RandomizedSearchCV, 60개 조합 × 5-fold CV. **`n_jobs`는 반드시 `1`** — Windows에서
   사용자 경로에 한글이 포함돼 있어 `n_jobs=-1` 사용 시 joblib 멀티프로세싱이 `UnicodeEncodeError`로
   실패함(`src/09_tune_xgboost.py`, `src/10_tune_random_forest.py`에 반영됨).

4. **Feature Importance 분석**: 완료 (`src/12_feature_importance.py` →
   `figures/feature_importance_xgboost.png`) — 최종 채택 모델(XGBoost 튜닝) 기준, gain importance +
   permutation importance 교차 확인.

   | 순위 | 변수 | Permutation Importance |
   |---|---|---|
   | 1 | **yr (연도)** | 0.507 |
   | 2 | **temp (기온)** | 0.345 |
   | 3 | hum (습도) | 0.103 |
   | 4 이하 | season, windspeed, mnth, weathersit, atemp, workingday, weekday, holiday | 낮음 |

   `yr`이 압도적 1위 — 날씨보다 "연도별 서비스 성장 추세"(2011→2012 이용자 기반 확대)가 예측에 더
   크게 기여. `hum`이 `season`보다 중요하게 나온 것은 계절 정보가 mnth/weathersit/temp에 분산돼
   개별 기여도가 낮아진 것으로 해석.

5. **이상치 탐색 (보너스)**: 완료 (`src/13_isolation_forest_outliers.py`) — Isolation Forest
   (contamination=5%)로 731일 중 37일(5.1%)을 이상치로 탐지. 중앙값 치환(median ablation) 방식으로
   각 날의 직접 원인 변수를 특정: **날씨등급 24일(64.9%) > 풍속 8일 > 체감온도 3일 > 기온 1일 >
   대여수 1일**. 상위 5개는 외부 검색으로 실제 기상 이벤트(겨울 폭풍, 기록적 폭우, 강풍, 허리케인
   샌디)와 대조 확인됨. 결과표: `tables/isolation_forest_outliers.csv`, 대시보드:
   `docs/outlier_dashboard.html`.

6. **예측 오차 분석 (보너스)**: 완료 (`src/14_prediction_error_analysis.py`) — 기준 모델로 731일
   전체를 예측해 오차 최대 10일을 추출하고, 이상치 37일 목록과 교집합 확인. **겹치는 날은
   2012-10-29(허리케인 샌디) 단 1건뿐** — 나머지 9건은 날씨는 평범하지만 특정 공휴일(추수감사절,
   독립기념일, 크리스마스이브 등)이라 `holiday` 이진 변수만으로는 설명 안 되는, 이상치 탐지로는
   못 잡는 모델의 별도 약점. 결과표: `tables/worst10_predictions.csv`.

   다음 후보(README "다음 단계" 절): 시간순 검증(TimeSeriesSplit), naive baseline 비교, 잔차 진단,
   시드 안정성 검증.

## 모델링 규칙

- 타겟은 `cnt`. **`casual`, `registered`는 설명변수에서 반드시 제외** — `cnt = casual + registered`
  이므로 포함 시 데이터 누수(R²=1.0으로 왜곡) 발생.
- train/test split은 `test_size=0.2, random_state=42`로 통일(모델 간 성능 비교 시 동일 split 사용).
  현재 분할은 무작위이며, 일별 시계열 자기상관을 고려하면 성능이 낙관적으로 부풀려졌을 가능성이
  있음(README "한계" 절 — 다음 단계로 TimeSeriesSplit 검증 예정).
- 트리 기반 모델(RandomForest, XGBoost)은 원-핫 인코딩 없이 순서형 인코딩 그대로 사용.
- 하이퍼파라미터 탐색은 RandomizedSearchCV(5-fold CV), `n_iter=60`. **`n_jobs`는 반드시 `1`** — 이
  프로젝트 경로에 한글(사용자명)이 포함돼 있어 병렬 처리(`n_jobs=-1`) 시 joblib이
  `UnicodeEncodeError`로 실패함.
- **pandas DataFrame을 만들 때 정렬된 Series와 정렬 안 된 numpy 배열을 섞지 말 것** —
  `pd.DataFrame({"a": some_sorted_series, "b": some_raw_array})`처럼 만들면 배열이 Series의 정렬된
  인덱스 순서에 맞춰 잘못 배치되는 버그가 발생함(실제로 feature importance 분석에서 발생, 차트
  라벨-값 불일치로 발견). 여러 소스를 합칠 때는 전부 동일한 인덱스를 명시할 것.
- 이상치·이례적인 날은 "통계적으로 튀는 값"이 아니라 "물리적으로 불가능한 값" 기준으로 판단(hum=0
  처리 사례). Isolation Forest 등 다변량 이상치 탐지 결과와 단변량 예측 오차는 서로 다른 것을
  포착하므로 혼동하지 말 것 — 5번·6번 항목 참고.

## 작업 규칙

- 새로운 분석/작업을 시작하거나 마칠 때 `docs/WORKLOG.md`에 기록한다. 최신 항목을 파일 맨 위에
  추가하고, 상단만 봐도 직전/다음 작업을 알 수 있도록 요약한다.
- 원본 데이터(`data/day.csv`)는 절대 수정하지 않는다. 가공 데이터가 필요하면 별도 파일로 `data/`에
  저장한다(`day_processed.csv` 패턴).
- 각 스크립트는 `data/`, `figures/`, `tables/`, `models/`를 상대경로로 참조하므로 **`01-washington/`
  루트에서** 실행해야 한다(`python src/01_data_overview.py` 등).
- 새 스크립트는 기존 번호 순서(현재 14번까지)를 이어서 작성하고, README의 스크립트 표·프로젝트
  구조를 함께 갱신한다.
