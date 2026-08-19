# 작업 로그 (원본 프로젝트에서 가져옴)

> 이 로그는 실제 분석을 진행하며 Claude Code와 함께 세션마다 작성한 작업 기록입니다.
> 최종 결과물뿐 아니라 **시행착오와 의사결정 과정**(과적합 발견, 버그 수정, 튜닝 실패 사례 등)을
> 그대로 남겨, 분석이 어떻게 진행됐는지 투명하게 보여주기 위해 가공 없이 포함했습니다.

---

## 2026-08-19 (최신 — 보너스 분석 2건, 프로젝트 재구조화, 시간별 데이터 검증)

- **Isolation Forest 이상치 탐지** (`src/13_isolation_forest_outliers.py`): 6개 변수(기온·체감온도·
  습도·풍속·날씨등급·대여수)로 731일 중 37일(5.1%, contamination=5%)을 이상치로 탐지.
  - 각 이상치의 직접 원인 변수는 **중앙값 치환(median ablation)** 방식으로 특정: 변수를 하나씩
    데이터셋 중앙값으로 바꿔치기했을 때 이상 점수가 가장 크게 정상 쪽으로 회복되는 변수를 원인으로
    판정. 결과: 날씨등급 24일(64.9%) > 풍속 8일 > 체감온도 3일 > 기온 1일 = 대여수 1일. 습도는
    항상 2위로만 등장(날씨등급과 상관돼 있어 1위를 뺏김).
  - 상위 5개 이상치(2011-01-26, 2011-10-29, 2012-12-26, 2011-02-19, 2012-10-29)를 웹 검색으로
    실제 기상 기록과 대조 — 전부 실제 이벤트와 일치(겨울 폭풍, 기록적 폭우, 강풍, **허리케인 샌디**).
    2012-10-29(대여수 22건, 전체 최저)는 허리케인 샌디가 D.C.를 강타해 대중교통이 이틀간 전면
    중단된 날짜와 정확히 겹침.
  - 결과표 `tables/isolation_forest_outliers.csv`, 인터랙티브 대시보드 `docs/outlier_dashboard.html`
    (Artifact로 제작 후 프로젝트에 저장).
- **휴일(holiday) 영향 분석** (일회성, 스크립트 미저장): 휴일 21일 vs 원래 근무일이었을 평일
  500일 비교. 전체 대여수 차이는 통계적으로 유의하지 않음(p=0.083)이지만, **회원(-32.9%, p=0.0007)과
  비회원(+75.5%, p=0.024)이 정반대로 움직이며 서로 상쇄**됨을 확인 — "휴일 효과 없음"이 아니라
  "이용자 구성이 뒤바뀌는 것"이 정확한 해석.
- **예측 오차 분석** (`src/14_prediction_error_analysis.py`): 기준 모델(XGBoost 튜닝)로 731일 전체
  예측 → 실제값과의 차이 최대 10일 추출, 이상치 37일 목록과 교집합 확인.
  - **겹치는 날은 2012-10-29(허리케인 샌디) 단 1건뿐.** 나머지 9건(2011-11-24 추수감사절,
    2012-07-04 독립기념일, 2011-12-24 크리스마스이브 등)은 날씨 입력값은 평범했지만 예측이 크게
    빗나감 — `holiday` 이진 변수로는 "어떤 공휴일인지"를 구분 못 해서 생기는, 이상치 탐지로는
    못 잡는 모델의 별도 약점으로 판단.
  - 이상치에만 있는 36건은 대부분 모델이 잘 예측함 — 입력이 특이하다고 항상 예측이 나빠지는 건
    아님을 보여줌.
  - 결과표 `tables/worst10_predictions.csv`.
- **README 갱신**: "이상치 탐색", "예측 오차 분석" 두 섹션과 스크립트 표(13, 14번) 추가.
- **프로젝트 재구조화**: 신규 데이터셋(런던) 분석을 위해 저장소 루트(`20th/`)를 도시별 폴더로 분리.
  기존 워싱턴 분석 전체(`data/`, `docs/`, `figures/`, `models/`, `src/`, `tables/`, `README.md`,
  `requirements.txt`)를 `git mv`로 **`01-washington/`** 하위로 이동(히스토리 보존), 빈
  **`02-london/`** 폴더 신규 생성.
- **CLAUDE.md 신규 작성** (`01-washington/CLAUDE.md`): 이전 세션(19th)의 CLAUDE.md를 참고 후보로
  검토했으나 폴더 구조·모델 결론이 실제와 달라(예: "XGBoost가 3개 지표 모두 1위"라고 잘못 적혀
  있었음) 그대로 쓰지 않고 새로 작성. ⚠️ **작성 중 오류 발견 및 정정**: 초안에서 "홀드아웃 MAE
  1위는 RandomForest(튜닝)"이라고 썼으나 `tables/model_comparison_final.csv` 원본 수치를 직접
  대조한 결과 실제 1위는 **RandomForest(default, 426.19)** 이고 튜닝된 RandomForest(436.56)는
  오히려 더 나쁨을 확인 — 바로잡아 반영함.
- **시간별 데이터셋(`bike_sharing_hour.csv`) 발견 및 검증**: `data/` 폴더에 시간별 원본이 추가돼
  day.csv와의 관계를 전부 실측으로 확인.
  - 컬럼: `hr`(시간대) 하나만 시간별에 더 있고 나머지 16개 컬럼은 동일. 기간도 2011-01-01~
    2012-12-31로 동일.
  - **day.csv는 hour.csv를 날짜별로 집계한 것임을 검증** — 날짜별 `cnt` 합산이 731일 전부
    day.csv와 정확히 일치(상관계수 1.0).
  - 시간별 행 수(17,379)가 이론값(731일×24시간=17,544)보다 **165건 부족** — 76일에서 결측 발생,
    최악은 2012-10-29(23시간 결측, 허리케인 샌디)·2011-01-27(16시간)·2012-10-30(13시간)·
    2011-01-18(12시간)·2011-01-26(8시간, 겨울 폭풍) — 이상치 분석에서 찾은 이벤트와 교차 일치.
  - **컬럼별 집계 방식을 6가지 후보(합계/평균/최대/최소/최빈값/하루내내동일)로 실측 대조**:
    season/yr/mnth/holiday/weekday/workingday는 하루 내내 동일값(731/731), temp/atemp/hum/
    windspeed는 평균(731/731), casual/registered/cnt는 합계(731/731). **weathersit만 예외** —
    6가지 중 어느 것도 완전히 안 맞음(최빈값이 632/731로 최선).
- **weathersit 집계 규칙 확정(사용자 정의, 미실행)**: 가장 나쁜 등급이 연속 4시간 이상 지속 →
  그 값, 아니면 최빈값, 동률이면 더 나쁜 쪽 우선. 아직 실제 데이터에는 적용하지 않음(기억만 해둔
  상태).
- **시간별→일별 변환 규칙서 작성** (`docs/HOUR_TO_DAY_AGGREGATION_RULES.md`): 위 검증 결과를
  컬럼 이름이 아니라 역할(유형) 기준으로 일반화해서 정리 — 런던 등 다른 데이터셋에도 재사용
  가능하도록 "새 데이터셋 적용 체크리스트" 포함. 작성 전 결측 시간 처리 방식·지속 기준·동률 처리·
  반올림 자릿수 4가지를 사용자에게 확인받고 작성.
- **다음 작업 후보**: London 데이터셋 전처리 시작(`02-london/`), weathersit 확정 규칙을 실제
  워싱턴 데이터에 적용해보기, 시간순 검증(TimeSeriesSplit) — 아직 미착수.

---

## 2026-08-14 (외부 리뷰 반영: 결론 정정)

- **직전 작업**: 제삼자 관점의 코드/분석 리뷰를 받아 **결론 서술의 오류를 정정**함
  - ❌ 기존 결론 "XGBoost(튜닝)이 3개 지표(R²/RMSE/MAE) 모두에서 1위" → **사실이 아님**.
    같은 홀드아웃에서 **MAE는 RandomForest(426.19)가 XGBoost 튜닝(435.14)보다 우수**했다.
    CV 프로토콜 한쪽만 보고 결론을 낸 것이 원인.
  - ❌ 상위 4개 모델의 CV R² 차이(0.872~0.893)를 순위로 단정 → **fold 표준편차(±0.024~0.026)보다
    작아 통계적으로 확정적이지 않음**. 특히 RF default(0.875) vs RF 튜닝(0.874) 차이 0.001은 노이즈.
  - ⚠️ **선택 편향 발견**: 튜닝 모델의 CV 점수는 하이퍼파라미터를 고른 구간과 채점 구간이 겹쳐
    낙관적으로 편향. default 모델과 나란히 순위를 매기는 것은 불공정한 비교였음.
  - ⚠️ **라벨 오류 정정**: `RandomForest (default)` / `XGBoost (default)`로 표기했던 모델은
    실제로는 `n_estimators=300`을 지정한 설정(XGBoost 기본값은 100) → `(n_est=300)`으로 수정.
- `src/11_final_model_comparison.py` 재작성: 하드코딩된 수치를 제거하고 원본 데이터에서 매번
  재계산하도록 변경. CV 표준편차 + 홀드아웃 지표를 함께 산출 → 그림도 2행(CV / 홀드아웃) 구성으로 교체.
  - 재계산 결과 기존에 보고한 CV 평균값(0.893/0.875/0.874/0.872/0.821/0.783)은 모두 그대로 재현됨.
- README 재작성: 분석 판단의 주어를 작업자 본인으로 정리하고, Claude Code 활용은 '작업 방식' 절로 축약.
  결론의 실무적 함의와 한계(특히 `yr` 외삽 불가)를 명시.
- **다음 작업 후보**: 시간순 검증(`TimeSeriesSplit` 또는 2012 하반기 홀드아웃) — 무작위 분할이
  자기상관 때문에 성능을 부풀리고 있는지 확인하는 것이 최우선. 이어서 naive baseline 비교, 잔차 진단.

---

## 2026-08-14

- **직전 작업**: 최종 채택 모델(XGBoost 튜닝) feature importance 분석 (`scripts/feature_importance.py` → `outputs/figures/feature_importance_xgboost.png`, `outputs/tables/feature_importance_xgboost.csv`)
  - gain importance(모델 내장) + permutation importance(test셋 R² 하락폭) 2가지로 교차 확인
  - **중요도 순위**: yr(0.507) > temp(0.345) > hum(0.103) > season(0.066) > windspeed(0.035) > mnth(0.027) > weathersit(0.021) > atemp(0.017) > workingday/weekday/holiday(0.003~0.005)
  - **`yr`(연도)이 압도적 1위** — 날씨보다 "연도별 서비스 성장 추세"가 예측에 더 크게 기여 (직관과 다른 결과라 사용자에게 별도 설명함)
  - ⚠️ **버그 수정 이력**: 최초 스크립트에서 정렬된 Series(gain_imp)와 정렬 안 된 numpy 배열(perm.importances_mean)을 한 DataFrame에 섞어 넣어 인덱스가 밀리는 버그 발생 → 차트 라벨-값 불일치로 발견 → `pd.Series(..., index=BASE_FEATURES)`로 명시적 인덱스 지정하여 수정. **앞으로 Series/array를 섞어 DataFrame 만들 때는 반드시 양쪽 다 동일 인덱스를 명시할 것**
- 최종 모델 성능 비교 시각화 완료 (`scripts/visualize_model_comparison.py` → `outputs/figures/model_performance_comparison.png`) — 6개 모델 CV R²/RMSE/MAE 비교, XGBoost(튜닝) 1위 확인
- 하이퍼파라미터 튜닝 완료: XGBoost는 튜닝으로 개선(CV R² 0.872→0.893), RandomForest는 튜닝해도 개선 없음(0.875→0.874, default 유지) — `scripts/tune_xgboost.py`, `scripts/tune_random_forest.py`
- **최종 모델 순위 (CV R² 기준)**: XGBoost(튜닝) 0.893 > RandomForest(default) 0.875 ≈ RandomForest(튜닝) 0.874 > XGBoost(default) 0.872 > LinearRegression(원핫) 0.821 > LinearRegression(순서형) 0.783
- **다음 작업 후보**: temp-atemp 다중공선성(VIF) 확인, 예측값 vs 실제값 잔차 플롯 — 아직 미정, 사용자 확인 필요

---

### 2026-08-14 (회귀모델 1차 비교 및 검증)

- 4개 모델 과적합 확인 + 5-Fold 교차검증 (`scripts/validate_models.py` → `outputs/tables/model_validation.csv`)
  - RandomForest: train R²=0.982 vs test R²=0.886 (gap=0.096) / CV R²=0.875±0.026
  - XGBoost: train R²=1.000 vs test R²=0.884 (gap=0.116, 뚜렷한 과적합) / CV R²=0.872±0.024
  - LinearRegression(원-핫): train 0.848 vs test 0.841 (gap=0.007) / CV R²=0.821±0.034
  - LinearRegression(순서형): train 0.792 vs test 0.828 (gap=-0.037) / CV R²=0.783±0.043
- 회귀 모델 4종 1차 비교 완료 (`scripts/compare_regression_models.py`) — 결과표 `outputs/tables/model_comparison.csv`, 모델 파일 `outputs/models/*.joblib`
- (참고용) casual/registered 포함 전체 컬럼 모델은 R²=1.000 — 데이터 누수 확인용으로만 실행, 실사용 불가 (`scripts/linear_regression_cnt_allcols.py`)

---

### 2026-08-14 (EDA·시각화)

- 이상치 처리: `scripts/clean_hum_outliers.py`로 hum=0(instant 69, 2011-03-10) 결측 처리 후 시간축 선형보간(→0.712) → `inputs/day_processed.csv` 생성 (원본 `day.csv`는 미수정)
- windspeed IQR 이상치 13건 레버리지 체크: 단순 OLS(cnt~temp+hum+windspeed) 기준 threshold(2k/n=0.0109) 대비 12/13건이 high-leverage로 나타남 (특히 instant 50, leverage=0.031) — 값 자체는 삭제하지 않고 유지
- 계절x요일 히트맵(`outputs/figures/season_weekday_heatmap.png`), EDA 시각화 3종(`scripts/visualize_eda.py`) 생성 완료
- 기술통계/이상치/결측치 확인 완료 (731행 16열, 결측치 0건)

---

### 2026-08-14 (초기 설정)

- 프로젝트 폴더 구조 생성: `inputs/`, `outputs/` (`figures/`, `tables/` 하위 포함), `docs/`
- 기존 `day.csv`(UCI Bike Sharing 일별 데이터셋)를 `inputs/`로 이동
- `CLAUDE.md` 작성: 프로젝트 개요, 폴더 구조, 데이터셋 컬럼 설명, 향후 진행 계획(EDA → 시각화 → 회귀분석) 정리

---

> 원본 경로에서는 `inputs/`, `outputs/figures/`, `outputs/tables/`, `outputs/models/`로 표기되어 있으나,
> 이 포트폴리오 저장소에서는 각각 `data/`, `figures/`, `tables/`, `models/`로 재구성했습니다.
