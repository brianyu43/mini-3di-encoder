# 후속 연구 실행 기록

시작: 2026-09-29, 기준 커밋 `b9de57f`, 브랜치 `research/hard-negative-evaluation`.
목표는 NEXT_RESEARCH.md를 순서대로 실행하는 것이다. 완료된 1–15회차의 원본 결과·모델·
기하 코드는 보존하고, 새 test를 보기 전에 데이터·규칙·비교 설정을 고정한다.

## R1: 어려운 음성을 포함한 평가

- 후보는 기존 SCOPe40 적격 catalog를 재사용한다. 학습은 기존 256개, 표현/행렬도 이미 선택된
  세 비교군으로 고정한다. 이번 R1에서는 재학습으로 결과를 바꾸지 않는다.
- 이전 test의 fold 중 b.34/c.26/d.17을 새 검색 validation에 쓴다. 이전 test는 이제 개발 자료다.
  새 test는 기존 384개 전체의 48 fold 밖에서 선정한다. query 100개, query SF ≥30개,
  target ≤1,000개를 목표로 한다. 구조 적격성이 모자라면 점수를 보기 전에 이유를 적는다.
- 모든 query에 같은 SF 양성, 같은 fold·다른 SF 음성, 다른 fold 음성을 확보한다. target에
  다른 family의 같은 SF 양성을 최소 하나 포함시킨다. 같은 family 양성은 별도 표시한다.
- 공식 공개 train/validation 목록의 정확한 SID 및 PDB membership을 각각 기록한다.
  둘 다 없는 query는 별도 층으로 보고하되 배포 가중치의 비노출을 증명한다고 하지 않는다.
- split 간 fold/SF/PDB/서열/구조 hash 겹침을 검사한다. 근접 서열은 global alignment에서
  identity ≥0.8 및 양쪽 aligned coverage ≥0.9로 정의한다. 근접 구조는 양쪽 길이 정규화
  TM-score의 최솟값 ≥0.95로 정의한다. 이 임계값은 검색 결과와 무관하게 사전 고정한다.
  구조 검사의 길이비 <0.95 쌍은 이 임계값에 도달할 수 없어 계산을 생략한다. 나머지 모든
  split 간 쌍을 TM-align으로 확인한다. 학습·개발에 노출된 구조도 test 중복 audit에 포함한다.
- primary는 같은 SF MAP/Recall@10, 보조는 같은-family target을 제외한 다른-family 검색,
  같은-fold 음성만 남긴 검색, PR 곡선, 공식 목록 비중복 층이다. 음성 층의 생물학적 의미를
  보존하고 같은 fold를 확정적 비상동성으로 취급하지 않는다.
- 공통 gap=(10,1)과 validation 선택 gap 후보 (6,1)/(10,1)/(14,2)를 분리한다.
  동점은 점수 내림차순, ID 오름차순. bootstrap은 fold cluster 2,000회다.
- warm Numba 점수 계산 10억 cell 예비 실행 후 총 CPU 2시간/4GiB 상한을 확인한다.
  데이터 동결 뒤에 test를 한 번 평가한다. 시간 반복은 같은 결과의 계산 비용만 재측정한다.

## 다음 단계의 판단

R1 완료 후 공통 gap에서 재학습 표현의 MAP 저하와 query 오류가 나타나면 R2의 학습량·상태
안정성 실험으로 진행한다. 후보·상세 정렬 비용이 여전히 지배하면 R3의 시스템 실험 근거도
기록한다. R2/R3 세부 설정과 test 사용 범위는 각 실행 전에 별도 동결한다. 결과를 확인한
데이터를 다시 새로운 독립 test라고 부르지 않는다. GPU·클라우드·큰 DB 다운로드는 포함하지 않는다.

| 작업 | 현재 상태 |
|---|---|
| R1 후보·분류층 구성 | 완료: test 100×484, 56 query SF / 34 fold |
| R1 근접 중복 audit·데이터 동결 | 완료: 46,906서열/22,861구조 비교, flag 0 |
| R1 비용 예비 측정·설정 동결 | 완료: 고정된 common/selected gap, 원본 freeze 보존 |
| R1 전수검색·오류/불확실성 분석 | 완료: 6,000 query/view/run 검산; paired fold 구간 추가 |
| R2 학습량·경계 안정성 | 완료: 15학습, 6좌표 조건, 78점수표 독립 검산 |
| R3 동일 결과 CPU 최적화 | 완료: 10규모·길이 조건, 1,000회 측정·저장 결과 검산 |
| 후속 계획·최종 보고 | 완료: NEXT_RESEARCH_AFTER_FOLLOWUP.md, FOLLOWUP_REPORT.md |

실제 출력은 `artifacts/followup-v2/`에 보존한다. R1–R3의 실제 완료와 명시된 표본 부족 조건을
근거로 후속 연구를 닫았다. 새 미실행 계획은 완료 결과와 구분한다.

## 실행 기록

- 데이터 후보: test 100query/484target, query SF 56개, 34fold. validation 30query/84target.
- 초기 후보에서 이전 자료와 같은 PDB인 항목을 찾아, 점수 계산 전에 제외해 다시 구성했다.
  첫 후보·구조는 `artifacts/followup-preselection-v1`에 보존한다. 이후 최종 후보를 동결했다.
- 비교할 314,760쌍 중 길이로 가능한 서열 46,906쌍/구조 22,861쌍의 근접 중복 audit가 완료됐다.
  flag 0개이며 전체 journal·TM-align 표본 원본 출력을 보존했다.
- 평가 분모·중립 target 제외·fold 가중 bootstrap 테스트를 추가했고 총 62개 테스트가 통과했다.
- R1과 조건부 R2 사전 등록을 `b313233`에 커밋했다. 당시 R2는 실행 전이었다.
- 첫 검색 준비에서 기존 엔진의 호출당 DP 예산 상한(10억 cell)을 확인했다. 실제 지표를
  계산하기 전의 실패 로그를 보존하고, 엔진을 수정하지 않고 query를 묶음으로 나누도록 했다.
  모든 query-target 쌍과 순서는 유지하며 결과를 합친다. 묶음 경계·전체 포함 테스트를 추가했다.
- R1 첫 test의 결과 기록에서 hash 함수 이름 충돌로 중단됐다. 원본 freeze·실패 기록을
  보존하고 `execution-repair.json`으로 수정 전후 코드 hash를 연결했다. 지표 저장 전 실패이며
  모델·자료·gap·평가 정의를 바꾸지 않았다. 수정 후 세 반복과 독립 검산이 통과했다.
- R1 learned−official_refit MAP는 −0.0074, paired fold 95% 구간 [−0.0255, +0.0087]이다.
  큰 하락 gate는 미충족이지만 AP<0.5 질의 22개가 있어 사전 등록한 R2로 진행했다.
- R2 38/76/152 구조 쌍 × 5seed를 모두 실행했다. test MAP 평균 .7258/.7372/.7287,
  full−quarter는 2seed 양수/3seed 음수다. 공식·학습 모델의 soft seed 모두 validation
  retain@10 .98 미달로 채택하지 않았다. 전체 모델·실패 사례·점수 원본을 보존했다.
- R2 한 학습 입력에서 Torch와 배포형 NumPy의 상태가 달랐다. BN folding/float32의
  약 2.5e−7 잠재 좌표 차이가 중심 경계를 넘었다. 행렬과 검색은 배포 상태를 일관되게 쓴다.
- R3는 기존 배포 모델·행렬·gap을 그대로 고정했다. 원본 검색 저장소를 수정하지 않고
  실험 커널을 별도로 구현했다. `85c7402`에 사전 계획·동등성 테스트를 기록했다.
- 테스트는 README의 `python -m pytest` 명령을 쓴다. 직접 `pytest`를 호출했을 때
  저장소 루트의 experiments import가 실패했던 실행은 테스트 성공으로 세지 않았다.
- R2 평가의 첫 실행은 검색 저장소 import 경로 설정 전에 멈췄다. 파일 동결·점수 생성 전
  실패이며 경로 설정 후 정상 실행했다. 실제 실행 코드 hash/커밋은 REPRODUCE.md에 있다.
- R3에서 50/200/1,000/5,000개 네 규모 모두 동일 결과의 backend 개선이 1.5배를 넘었다.
  5,000개에서 12.524→1.674초(7.48배), 필터 retain@10=.94다. 필터 자체를 높은 보존율의
  기본값으로 채택하지 않는다. 길이별 5,000개는 각각 3,514/4,646/2,094개만 적격이라 생략했다.
- R3 전체 1,000개 timed run의 후보·점수 hash와 중앙값/범위를 검산하고 원본 경로를 재채점했다.
  프로세스 peak RSS 553.4MB, GPU·클라우드·새 구조 다운로드 없음. 원본 검색 저장소는 그대로다.

## 최종 검증

- research 환경: `PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python -m pytest -q`
  → **71 passed**. 원본 source 비교·새 커널의 300정렬/35후보 무작위 사례를 포함한다.
- 기본 인코더 환경: `.venv/bin/python -m pytest -q` → **63 passed, 3 skipped**.
  PyTorch/Numba가 필요한 선택적 연구 모듈을 건너뛰고 기본 기능을 검증했다.
- `ruff check .`, `ruff format --check .` 통과. 생성 SVG의 불필요한 줄 끝 공백을
  정리한 뒤 `git diff --check`도 통과했다. 그림 데이터와 시각적 내용은 바뀌지 않았다.
- 공식 모델 및 새 full-data seed71로 실제 1UBQ CLI 실행 성공: 각각 76잔기, 유효 특징 74개.
  결과는 `artifacts/followup-v2/final-example-official` / `final-example-learned`에 있다.
- R1·R2·R3 그림을 직접 확인했고 결과표·분모·hash·실행 코드 커밋을 대조했다.
- 연구 브랜치에 코드·모델·표·그림을 로컬 커밋했다. 새 GitHub push는 하지 않았다.
  다음 연구의 구체적 계획은 미실행 상태로 분리했다.
