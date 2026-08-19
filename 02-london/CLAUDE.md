# 프로젝트 개요

런던 자전거 대여 데이터셋(Kaggle "London bike sharing dataset", 시간별)을 일별로 변환한 뒤,
**2015년 데이터로 회귀모델을 학습해 2016년 대여 건수(`cnt`)를 예측**하는 프로젝트. 저장소 루트
(`20th/`)는 도시별 분석을 나란히 담는 컨테이너이며, 이 폴더(`02-london/`)는 그중 런던 데이터를
담고 있다. 같은 루트의 `01-washington/`은 워싱턴 D.C. 데이터로 EDA·시각화·회귀분석까지 완료된
별도 프로젝트로, 이 프로젝트의 워크플로우·규칙서를 재사용/재검증하며 진행했다.

**현재 진행 상태: 전처리 → EDA → 회귀분석(모델비교·튜닝·최종비교) → 변수중요도 → 이상치탐지 →
예측오차분석 → 리포트까지 전 과정 완료.** 결과 요약은 [`README.md`](README.md), 상세 근거는
[`docs/REPORT.md`](docs/REPORT.md) 참고.

## 폴더 구조

```
02-london/
├── data/
│   ├── london_merged.csv  # 원본 시간별 데이터 (절대 직접 수정하지 않음)
│   ├── day.csv             # 일별 변환본 (전체 기간, 730행, 16열)
│   ├── day_2015.csv        # train (362행, 2015-01-04~12-31)
│   └── day_2016.csv        # test/홀드아웃 (365행, 2016-01-01~12-31)
├── src/                     # 01~14, 숫자 순서대로 실행 (README.md 스크립트 표 참고)
├── figures/                 # 시각화 (png)
├── tables/                  # 결과표 (csv)
├── models/                  # 학습된 모델 (joblib)
├── docs/
│   ├── ANALYSIS_PLAN.md    # 통합 분석 계획서 (컬럼 설계 근거 포함)
│   ├── REPORT.md           # 상세 분석 리포트
│   └── WORKLOG.md          # 작업 이력 로그
├── requirements.txt
└── CLAUDE.md                # 이 파일. 프로젝트 컨텍스트 요약
```

- 스크립트는 `src/`에 작성. `02-london/` 루트에서 실행해야 상대경로(`data/...`)가 맞는다.
- git 저장소(`20th/` 루트에 `.git`)로 관리 중. 커밋은 사용자가 명시적으로 요청할 때만 진행.

## 데이터셋

### data/london_merged.csv (원본, 시간별)

17,414행, 결측치 0건, 기간 2015-01-04 00:00 ~ 2017-01-03 23:00 (1시간 단위).

| 컬럼 | 설명 |
|---|---|
| timestamp | 관측 시각 |
| cnt | 자전거 대여 건수 (타깃) |
| t1 | 실제 기온(°C) |
| t2 | 체감 기온(°C) |
| hum | 습도(%) |
| wind_speed | 풍속 |
| weather_code | 날씨 코드 — 1=맑음, 2=약간 흐림, 3=구름 많음, 4=흐림, 7=비, 10=뇌우, 26=눈. 숫자가 클수록 나쁜 날씨(실제 기온·습도 패턴으로 검증됨) |
| is_holiday | 공휴일 여부(0/1) |
| is_weekend | 주말 여부(0/1) |
| season | 계절 — 0=봄, 1=여름, 2=가을, 3=겨울(t1 평균·최빈 월로 검증됨) |

### data/day.csv, day_2015.csv, day_2016.csv (변환본, 일별)

`01-washington/docs/HOUR_TO_DAY_AGGREGATION_RULES.md`의 컬럼 유형별 규칙을 런던에 맞게 적용해
만들었다(근거는 `docs/ANALYSIS_PLAN.md`, 변환 로직은 `src/01_day_conversion.py`).

| 컬럼 | 생성 방식 |
|---|---|
| instant | 1부터 순번 재부여 |
| year, month, day | `timestamp`에서 추출 (실제 연/월/일 값) |
| weekday | 요일 이름 문자열(Monday~Sunday) |
| season, is_holiday, is_weekend | 하루 내내 동일한 원본 값 그대로 |
| working | 파생: `is_holiday==0 and is_weekend==0`이면 1 (모델 피처로는 미사용, 아래 참고) |
| weather_code | 3단계 규칙(연속4시간 지속 최악등급→최빈값→동률시 나쁜쪽)으로 산출한 하루 대표값 |
| t1, t2, hum, wind_speed | 하루 평균(소수 1자리) |
| cnt | 하루 합계 |
| n_hours_observed | 그날 실제 관측 시간 수(0~24) — 결측 많은 날 식별용 |

`casual`/`registered`(워싱턴에 있던 대여자 유형 구분)는 원본에 없어 만들지 않았다.

## 모델링에서 확정한 컬럼 처리 (재사용 시 반드시 참고)

일별 변환본에는 위 16개 컬럼이 다 있지만, **회귀모델 피처로는 아래 원칙을 따른다**(근거는
`docs/ANALYSIS_PLAN.md` 2절, `docs/REPORT.md` 2-2절):

- **`year` 완전 제외** — train(2015)/test(2016) 각각에서 상수라 학습 불가.
- **`working` 제외, `is_holiday`+`is_weekend` 사용** — `working`은 이 둘의 결정론적 함수(실측
  확인)라 셋을 함께 쓰면 다중공선성.
- **`weekday`**: 선형회귀=원-핫, 트리모델=달력 순서(월=0~일=6) 정수.
- **`t2`(체감기온)는 선형회귀 계열에서만 제외** — `t1`-`t2` VIF 88~91로 심각한 다중공선성
  확인(`src/04_multicollinearity_check.py`). 트리 모델은 둘 다 사용.
- **`n_hours_observed`는 피처로 쓰지 않고 진단용으로만** — 저신뢰 날짜(예: 2016-06-24, 9시간만
  관측)를 예측 오차 분석에서 원인 설명에 실제로 활용함.
- **`weather_code=26`(눈)은 train(2015)에 0건, test(2016)에 1건뿐** — 원-핫 인코딩 회귀에서
  이 범주의 계수를 전혀 학습하지 못함(한계로 명시).

## 최종 결과 요약

- **최종 채택 모델: LinearRegression(원-핫 인코딩)** — 2016 홀드아웃 R²=0.769, RMSE=4,189,
  MAE=3,120. CV 1위였던 XGBoost(튜닝, CV R²=0.793)는 홀드아웃에서 R²=0.749로 밀려남 — **CV
  1위 ≠ 홀드아웃 1위**가 실제로 재현된 사례(`tables/final_model_comparison.csv`).
- **변수 중요도(그룹 permutation importance)**: `t1`(기온) 0.517 압도적 1위, `hum` 0.181,
  `wind_speed` 0.101 순. `year`를 뺐기 때문에 워싱턴처럼 "연도가 1위"인 결과는 구조적으로 나올 수
  없음(`tables/feature_importance.csv`).
- **이상치(2015, Isolation Forest 5%)**: 19일. 최상위(2015-07-01, 원인=t1)는 실제 런던 히스로
  36.7°C(2015 UK 7월 최고기온 신기록)와 대조 확인.
- **예측오차 최상위(2016)**: 2016-12-25(오차 24,245) — 실제로 관측 사상 가장 따뜻한 크리스마스
  중 하나였음을 웹 검색으로 대조. 2위 2016-06-24(오차 18,115)는 저신뢰 날짜(9시간만 관측)라
  실제 예측 실패가 아님.

## 작업 규칙

- 원본 데이터(`data/london_merged.csv`)는 절대 수정하지 않는다.
- 새로운 작업을 시작하거나 마칠 때 `docs/WORKLOG.md`에 기록한다(최신 항목 상단, "직전 작업/다음
  작업" 두 줄 요약 패턴 유지).
- 컬럼 처리 방침을 바꾸면 `docs/ANALYSIS_PLAN.md`·`docs/REPORT.md`도 함께 갱신한다.
- 각 스크립트는 `data/`, `figures/`, `tables/`, `models/`를 상대경로로 참조하므로 **`02-london/`
  루트에서** 실행해야 한다. `n_jobs=1` 고정 필요(Windows 한글 경로로 인한 joblib 이슈, 워싱턴과
  동일).

## 다음 작업 후보

1. **포트폴리오 정리** (진행 중) — 저장소 루트 README 신설 + 두 프로젝트 폴더 정리.
2. naive baseline 비교, `weather_code=26` 데이터 부족 보완, 잔차 진단, 워싱턴과의 합동 비교 —
   `docs/REPORT.md` 8-2절 참고.
