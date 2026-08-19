# 작업 로그

> 최신 항목이 파일 맨 위에 오며, 각 날짜 항목 상단의 "직전 작업 / 다음 작업" 두 줄만 봐도 전체
> 흐름을 알 수 있게 정리합니다. 상세 내역(시행착오·중간 결정 포함)은 그 아래에 남깁니다.

---

## 2026-08-19 (최신 — 통합 계획 수립 및 회귀분석 파이프라인 전체 실행)

- **직전 작업**: `docs/ANALYSIS_PLAN.md`(2015→2016 예측 통합 계획) 수립 및 5가지 미해결 항목 확정
  → 계획대로 `src/02_data_overview.py`~`src/14_visualize_final_comparison.py`까지 13개 스크립트를
  순차 실행해 EDA·다중공선성 점검·모델 비교·튜닝·최종비교·변수중요도·이상치탐지·오차분석을
  전부 완료. `README.md`·`docs/REPORT.md` 신규 작성.
- **다음 작업 후보**: 포트폴리오 정리 — 저장소 루트 README 신설 + `01-washington`/`02-london` 폴더
  다듬기(사용자 요청, 진행 중). 이후 naive baseline 비교, `weather_code=26` 데이터 부족 보완,
  잔차 진단, 워싱턴과의 합동 비교가 분석 쪽 다음 후보.

### 상세 내역

- **통합 계획서 작성** (`docs/ANALYSIS_PLAN.md`): 워싱턴 14개 스크립트 워크플로우를 요약하고,
  런던이 워싱턴과 근본적으로 다른 지점(연도 단위 시간 분할 train/test, `year` 상수 문제,
  `working` 다중공선성)을 표로 정리. 실행 전 5가지 미해결 항목(year 제외 여부, working/is_holiday/
  is_weekend 선택, weekday 인코딩, n_hours_observed 처리, weather_code/season 인코딩 실험 범위)에
  대해 근거 있는 제안을 먼저 받고 전부 승인받아 계획서에 확정 반영.
- **02_data_overview.py**: train(2015)/test(2016) 간 season·is_holiday·is_weekend 분포가 고르게
  분산돼 있음을 확인. `weather_code=26`(눈)이 train엔 0건, test엔 1건뿐임을 발견(이후 원-핫
  회귀의 한계로 이어짐).
- **03_visualize_eda.py**: 온도-대여량 산점도, weather_code별 박스플롯, 계절×요일 히트맵,
  train/test cnt 분포 비교 4종 시각화. 한글 폰트(Malgun Gothic) 미설정으로 글자가 깨지는 문제를
  워싱턴 스크립트와 대조해 발견·수정.
- **04_multicollinearity_check.py**: `working == (is_holiday==0)&(is_weekend==0)`이 362행 전부
  성립함을 재확인. VIF 점검 결과 `t1`-`t2`(기온-체감기온) VIF가 88~91로 심각해, 이후 선형회귀
  계열에서는 `t2`를 제외하기로 결정(계획서에 없던 추가 판단, 근거를 스크립트 docstring에 명시).
- **05~07 (baseline, 모델 비교, 과적합 검증)**: 2016 홀드아웃을 전혀 건드리지 않고 2015 내부
  5-fold CV로만 평가. 기본 설정 XGBoost가 train R²=1.000, val R²=0.698(gap=0.302)로 심하게
  과적합됨을 확인.
- **08~09 (하이퍼파라미터 튜닝)**: RandomizedSearchCV(60회×5-fold, `n_jobs=1` 고정 — 워싱턴과
  동일한 한글 경로 이슈 회피). XGBoost CV R² 0.698→0.793, RandomForest 0.748→0.771로 개선.
- **10_final_model_comparison.py**: 튜닝된 모델을 2015 전체로 재학습해 2016 전체를 한 번만 예측.
  **CV 1위(XGBoost 튜닝, 0.793)와 실제 2016 홀드아웃 1위(LinearRegression 원-핫, 0.769)가
  다름을 확인** — 원-핫 선형회귀가 R²·RMSE·MAE 세 지표를 전부 석권. `weather_code=26`처럼
  한쪽 세트에만 있는 범주 때문에 원-핫 컬럼이 어긋나는 문제는 train+test를 합쳐서 인코딩한 뒤
  다시 분리하는 방식으로 해결.
- **11_feature_importance.py**: 최종 채택 모델이 원-핫 선형회귀로 나오면서, 애초에 트리 모델
  전용으로 짜뒀던 스크립트를 원-핫 구조에 맞게 다시 작성 — 같은 원본 변수에 속한 더미 컬럼들을
  함께 셔플하는 "그룹 permutation importance"를 새로 구현. 결과: `t1`(기온) 0.517로 압도적 1위,
  `hum`(0.181) > `wind_speed`(0.101) 순. `year`를 뺐기 때문에 워싱턴처럼 "연도가 1위"인 결과는
  구조적으로 나올 수 없음.
- **12_isolation_forest_outliers.py**: 2015년 362일 중 19일(5.2%) 이상치 탐지, 중앙값 치환으로
  원인 변수 특정(cnt 6일 > t1 5일 > t2 4일 순, 워싱턴처럼 날씨등급 하나에 몰리지 않고 분산됨).
  최상위 이상치 2015-07-01을 웹 검색으로 대조 — **런던 히스로 공항 36.7°C, 2015년 UK 7월 최고기온
  신기록일**과 정확히 일치함을 확인.
- **13_prediction_error_analysis.py**: 처음엔 트리/순서형 모델만 가정하고 짜서 원-핫 모델이
  1위가 되자 `SystemExit`로 막히도록 해뒀던 부분을 10번 스크립트와 동일한 원-핫 재구성 로직으로
  수정. 2016 예측 오차 상위 10일 중 최댓값은 2016-12-25(오차 24,245) — 웹 검색으로 **런던 관측
  사상 가장 따뜻한 크리스마스 중 하나(약 15°C)** 였음을 확인, 모델이 겨울+주말 하락 효과를
  과대적용한 것으로 해석. 2위는 2016-06-24(오차 18,115)인데, 이 날은 `n_hours_observed=9`로
  이미 전처리 단계에서 저신뢰로 표시해뒀던 날짜 — 실제 예측 실패가 아니라 데이터 결손이 원인임을
  교차 확인. 2016년 자체 이상치(19일)와 오차 상위 10일의 교집합은 1건(2016-09-03)뿐.
- **14_visualize_final_comparison.py**: CV vs 홀드아웃 R² 비교 + RMSE/MAE 비교 그림 생성
  (`figures/model_performance_comparison.png`), 워싱턴 README의 대표 그림과 같은 역할.
- **README.md, docs/REPORT.md 신규 작성**: 워싱턴과 동일한 2단 구조(README=결과·결론·한계,
  REPORT=상세 근거)로 작성. requirements.txt도 신규 작성(seaborn, statsmodels 추가).
- **포트폴리오 작업 방향 확정** (실행은 다음 단계): GitHub에 "클로드코드로 데이터 분석을 할 수
  있다"는 것을 어필하는 포트폴리오로 올리고 싶다는 요청 → 저장소 루트에 신규 README(포트폴리오
  소개, 두 프로젝트 요약) + `01-washington`/`02-london` 폴더 정리로 범위를 확정. 런던 분석이
  끝난 뒤 진행하기로 결정(이 항목 완료로 이제 착수 가능).

### 상세 내역

- **원본 시간별 데이터 확인**: `data/london_merged.csv` 로드해 행/열, 컬럼별 자료형, 결측치,
  기간을 확인. 17,414행 × 10열, 결측치 0건, 기간 2015-01-04~2017-01-03(1시간 단위).
- **워싱턴 규칙서와 컬럼 대조** (`01-washington/docs/HOUR_TO_DAY_AGGREGATION_RULES.md` 기준):
  - 그대로 적용 가능: 연속형(t1/t2/hum/wind_speed→평균), 카운트형(cnt→합계), 날짜속성형
    (is_holiday→동일값).
  - 그대로 적용 불가: `weather_code`(워싱턴 weathersit과 값 체계 완전히 다름 — 1~4 연속 정수 vs
    1,2,3,4,7,10,26 비연속 코드), `season`(값 범위 다름, 워싱턴 1~4 vs 런던 0~3), `is_weekend`
    (워싱턴의 weekday+workingday 두 컬럼에 대응할 정보량이 부족한 한 컬럼).
  - 런던에 없어서 새로 만들어야 하는 컬럼: dteday, yr, mnth, weekday, workingday. `casual`/
    `registered`는 원본에 대여자 유형 구분 데이터 자체가 없어 생성 불가로 확정.
- **`season`/`weather_code` 정의 검증**: Kaggle 공개 코드북 기준 정의(season 0=봄·1=여름·2=가을·
  3=겨울, weather_code 1=맑음~26=눈)를 실제 데이터 패턴과 대조.
  - season: 계절별 t1 평균이 여름(18.4℃)>가을(13.0℃)>봄(10.7℃)>겨울(7.7℃) 순으로 상식과 일치,
    최빈 월도 봄=5월/여름=8월/가을=10월/겨울=1월로 일치 → 정의 신뢰 가능.
  - weather_code: 코드가 클수록 습도 평균이 대체로 증가(1=68.0%→26=88.8%), 눈(26)이 최저기온
    (5.2℃)·최고습도 → "숫자가 클수록 나쁜 날씨" 전제 유효. 단 코드10(뇌우)은 표본 14건뿐이라
    순서가 다소 흔들림(표본 부족으로 판단).
- **일별 집계 사전 검증**: `season`/`is_holiday`/`is_weekend`가 730일 전부 하루 내 완전히 동일함을
  확인(불일치 0건). 결측 시간 스캔 결과 34일에서 총 106시간 결측(최악 2016-06-24, 9시간만 기록).
- **1차 변환 스크립트 작성·실행** (`src/01_hour_to_day_aggregation.py`, 이후 삭제됨) →
  `data/day.csv`(구버전, dteday 방식) 생성. season/weather_code는 워싱턴의 weathersit 전용
  3단계 규칙(연속4시간 지속→최빈값→동률시 나쁜 쪽)을 코드 값 그대로 적용.
- **변환 계획서 작성** (`src/day_conversion_plan.md`): 사용자 요구사항(n_hours_observed 컬럼,
  timestamp에서 연/월/일 추출, working=주말도 휴일도 아닌 날 1, weather_code/season 원본 값
  그대로 활용) 반영해서 컬럼별 규칙을 문서화. 원본 파일은 절대 수정하지 않고 결과는 새 파일로
  저장한다는 원칙 명시.
- **미해결 항목 처리 방안 제안 및 확정**:
  - `weekday`: 추가하기로 결정. 숫자 인코딩(0=일요일 vs 0=월요일) 컨벤션 혼동을 피하려고 요일
    이름 문자열(Monday~Sunday)로 저장하기로 확정.
  - `casual`/`registered`: 계속 만들지 않기로 확정 — 빈 컬럼(NaN)으로라도 자리만 만드는 방안도
    검토했으나, 실제 값 없이 스키마만 맞추면 결측치 처리 혼동만 커진다고 판단.
- **최종 변환 스크립트 작성·실행** (`src/02_day_conversion.py`, 이후 `src/01_day_conversion.py`로
  이름 변경) → `data/day_v2.csv` 생성(instant, year, month, day, weekday, season, is_holiday,
  is_weekend, working, weather_code, t1, t2, hum, wind_speed, cnt, n_hours_observed, 730행 16열,
  결측치 0건).
- **검증**: `london_merged.csv`에서 직접 재계산해 대조 — 시간별 `cnt` 총합(19,905,972)과 일별
  `cnt` 총합 완전 일치, 임의 날짜 3건(2015-01-04, 2016-06-24 결측일, 2016-12-25)을 직접
  재계산해 `day_v2.csv` 값과 소수점까지 일치 확인.
- **파일 정리**: 구버전 `data/day.csv` 삭제 후 `data/day_v2.csv`를 `data/day.csv`로 이름 변경.
  더 이상 최신 계획을 반영하지 않는 구버전 스크립트 `src/01_hour_to_day_aggregation.py` 삭제(혼동
  방지). 남은 스크립트를 `src/02_day_conversion.py` → `src/01_day_conversion.py`로 재번호 부여,
  내부 출력 경로도 `data/day_v2.csv` → `data/day.csv`로 수정해 스크립트-산출물 이름을 일치시킴.
- **연도별 파일 분리**: `data/day.csv`를 연도별로 나눠 `data/day_2015.csv`(362행, 1/4~12/31),
  `data/day_2016.csv`(365행)로 저장. 2017년 3일치(1/1~1/3)는 제외.
- **문서화**: `CLAUDE.md`, `docs/WORKLOG.md`(이 파일) 신규 작성 — 워싱턴 프로젝트의 문서 구조를
  참고해 런던 프로젝트에 맞게 작성.

---
