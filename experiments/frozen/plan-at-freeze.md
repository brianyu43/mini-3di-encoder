# 11–15회차 실행 계획

요청: 남은 학습·평가 실험을 실제 완료하고, 결과에 근거한 다음 연구 계획까지 작성한다.
시작점: `a183bfa`, 작업 브랜치 `research/sessions-11-15`. 기존 공식 모델 재현 기능은 보존한다.
후속 연구는 구체적인 계획까지만 작성한다. 새 연구를 무한히 추가하지 않는다.

## 먼저 고정하는 실험 계약

- 데이터: 이미 내려받은 SCOPe40 PDB 묶음과 분류표. 과거 D1/D2·1UBQ의 도메인/PDB를 제외한다.
  단일 사슬, 길이 60–250, 표준 잔기 또는 UNK, 명시적 결측/단절 기준을 적용한다.
- 목표 규모: 서로 다른 fold에 속한 superfamily 48개에서 각 8도메인, 총 384개.
  train 32그룹/256개, validation 8그룹/64개, test 8그룹/64개.
  메타데이터 적격 수가 부족하면 학습 전에만 규모를 조정하고 이유를 기록한다.
  fold·superfamily·PDB·AA/구조 중복의 split 간 겹침을 0으로 검사한다.
- 선택: 고정 hash 순서와 a/b/c/d 클래스 균형. 결과를 본 뒤 구조를 교체하지 않는다.
  validation/test에서 그룹당 2query, 6target을 지정한다. 동일 PDB self-hit은 없다.
- 학습 쌍: train/validation 내부 같은 superfamily의 서로 다른 구조를 TM-align으로 정렬한다.
  그룹당 최대 12쌍을 결과와 무관한 hash로 선택한다. 양쪽 길이 정규화 TM-score의 최솟값
  0.6 이상, TM-align의 거리 5Å 미만 `:` 대응 중 양쪽 유효 위치만 사용하고 양방향으로 만든다.
  test의 구조 정렬·특징 쌍은 모델·행렬·설정 선택에 사용하지 않는다.
- 모델: 공식 공개 학습 소스의 10→10→10→2 encoder와 BatchNorm, 20상태 VQ,
  commitment 0.25, 대응 특징 Y에 대한 Gaussian NLL + VQ loss를 사용한다.
  CPU 1계산 스레드, Adam lr=0.001, batch=512, 최대 60epoch, seed 17/29/43.
  epoch별 validation loss로 체크포인트를 선택하고, 그 loss로 최종 seed를 고른다.
  모든 seed 결과와 상태 사용 빈도를 보존한다. test를 본 재학습은 하지 않는다.
- 새 행렬: train 대응 쌍의 대칭 상태 공동 빈도, 각 상태쌍에 pseudocount 0.5,
  half-bit log-odds, 정수 반올림, X 행/열=0. 공식 추정법과의 차이를 문서화한다.
- 비교군: 공식 알파벳+공식 행렬, 공식 알파벳+train에서 추정한 행렬,
  자체 학습 알파벳+같은 방식으로 추정한 행렬. 같은 검색 엔진/CPU backend를 쓴다.
- validation에서만 gap 후보 (6,1), (10,1), (14,2)를 비교하여 MAP 기준 선택한다.
  점수 스케일이 다른 알파벳끼리 raw score 크기를 직접 비교하지 않는다.
- 고정 평가: 먼저 전수검색, 그다음 k=3, window=64의 single/double/ungapped 필터.
  ungapped threshold=20. test 실행 전 선택 설정과 입력·모델 hash를 동결한다.
  MAP, Recall@1/5/10, 관련 대상 1위 비율, exact top10 보존율, 후보 수, DP cells,
  시간·메모리를 보고한다. query 대신 superfamily 단위 bootstrap 불확실성을 산출한다.
- 시간 측정은 같은 환경에서 반복하고 초기 JIT/입력/인코딩/검색을 분리한다.
  공개 공식 학습 목록과의 겹침을 조사하며, 확인되지 않은 일반화·새로움은 주장하지 않는다.

## 단계와 산출물

| 회차 | 통과 조건 | 상태 |
|---|---|---|
| 11 | 적격성·중복·분리 audit, 고정 manifest, 실제 구조와 출처 | 진행 중 |
| 12 | TM-align 원본 출력과 대응 index 검사, 실제 X/Y·mask, 제외 사유 | 대기 |
| 13 | 실제 역전파·체크포인트·loss·상태 사용률, NumPy 배포 모델 검증 | 대기 |
| 14 | train 행렬, validation 선택, 동결된 test 전수/필터 비교 | 대기 |
| 15 | 결과 재계산·실패 분석·그림·비용·재현 명령·후속 연구 계획 | 대기 |

실행 로그·입력·체크포인트는 `artifacts/sessions-11-15/`에 보존한다. 결과가 나쁘거나 상태가
붕괴해도 숨기지 않는다. 완료 조건은 공식 성능을 이기는 것이 아니라 검증 가능한 전체 실험이다.
