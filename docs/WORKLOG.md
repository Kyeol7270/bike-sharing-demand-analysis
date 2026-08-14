# 작업 로그 (원본 프로젝트에서 가져옴)

> 이 로그는 실제 분석을 진행하며 Claude Code와 함께 세션마다 작성한 작업 기록입니다.
> 최종 결과물뿐 아니라 **시행착오와 의사결정 과정**(과적합 발견, 버그 수정, 튜닝 실패 사례 등)을
> 그대로 남겨, 분석이 어떻게 진행됐는지 투명하게 보여주기 위해 가공 없이 포함했습니다.

---

## 2026-08-14 (최신)

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
