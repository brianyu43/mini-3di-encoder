# 회차별 진행 기록

## 현재 상태

- 실행 산출물 기준 **1 / 15회차 완료**.
- G1(한 잔기의 자체 인코딩 추적), G2(공식 일치·검색 연결), G3(재학습·평가)는 모두 아직 미통과.
- 사용자의 학습·이해 완료를 자동으로 판정하지 않았다. 아래 설명 질문을 다음 학습 시작점으로 사용한다.
- 다음 실행 범위: **2회차**, N/CA/C/CB 좌표와 원자 존재 상태 출력. 아직 시작하지 않았다.

## 2026-09-15 · 1회차

### 실행한 것

1. `Documents/dev/mini-3di-encoder`에 새 프로젝트·별도 Python 가상 환경을 만들었다.
2. 사용자의 3주 계획을 PLAN.md에 정리하고 단계별 통과 기준·데이터 분리·재학습 행렬 계약을 추가했다.
3. 기존 로컬 Foldseek 10-941cd33을 복사해 version 및 SHA-256을 확인했다.
4. RCSB의 1UBQ PDB 하나를 다운로드했다. 공식 구현·가중치·학습 코드 일부도 출처와 hash를 고정했다.
5. A 사슬의 잔기 번호·insertion code·AA·CA 좌표를 읽는 첫날용 Python reader를 작성했다.
6. 공식 `createdb`와 `convert2fasta`를 실행해 AA/3Di 원본 출력을 보관했다.
7. 좌표 표의 AA와 공식 AA의 정확한 일치, 3Di와 잔기 개수 일치, 원본 PDB 좌표와 표의 일치를 검증했다.

### 실제 결과

| 항목 | 확인 결과 |
|---|---|
| 구조 | 1UBQ, 모델 1, 사슬 A |
| 잔기 / 공식 AA / 공식 3Di 길이 | 76 / 76 / 76 |
| 자체 CA 표의 AA와 공식 AA | 76잔기 전체 정확히 일치 |
| 원본 PDB 고정 열과 CA 표 독립 대조 | 76개 좌표·잔기 번호·insertion code 일치 |
| 첫 잔기 | MET / M, PDB 번호 1, index 0, CA=(26.266,25.413,2.842) Å |
| 공식 도구 재실행 | AA FASTA·3Di FASTA·잔기 TSV가 byte 단위 동일 |
| Python | 3.12.14, macOS arm64 |
| 단위 검사 | **11 passed**; 누락 CA·altloc·비표준 잔기·다중 모델·번호 대응·헤더 대응·기존 출력 보호 |
| Ruff / format / pip check | 통과 |
| 외부 참고 파일 원본 hash | 16개 일치 |
| 오프라인 설치용 wheel hash | 9개 일치 |
| 자체 인코더 / 새 모델 학습 / 검색 비교 | 모두 미구현·미실행 |

검증 로그: [results/verification](results/verification/validation.json).
실제 명령: [results/day01/commands.json](results/day01/commands.json).
입력 SHA-256: `d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161`.

첫 성공 실행의 createdb wall time은 약 0.012초다. 프로세스 시작·출력 등을 포함한 한 구조의
단일 관찰이며 인코더의 일반적인 성능으로 주장하지 않는다. Python 프로세스 RSS high-water는
51,953,664 bytes, 종료된 자식 프로세스들에 대해 OS가 보고한 값은 13,238,272 bytes였다.
두 수치는 동시 전체 메모리 사용량이 아니며 합산하지 않는다. CPU 계산 thread 1개, GPU 미사용.

### 발생한 문제와 해결

- **공식 3Di DB에 FASTA 헤더가 없음:** 첫 `convert2fasta db_ss`가
  `Database ... needs header information`으로 실패했다. release 10은 AA 헤더 DB를 공유한다.
  AA/3Di/header의 내부 key 집합이 같음을 검사한 다음 헤더를 byte 단위 복사해 해결했다.
  key가 다른 경우 실패하는 테스트를 추가했다. 첫 실패 로그는 `artifacts/day01-header-attempt`에 보존했다.
- **참고 코드에 포맷터가 적용됨:** Git 초기화 전 첫 전체 Ruff 실행이 upstream Python 참고 파일까지
  변경했다. 고정 URL에서 원본을 다시 받아 manifest의 SHA-256과 대조해 모두 복원했다.
  Ruff의 `extend-exclude`에 외부 소스·tools·artifacts를 명시하고 최종 원본 hash 검증을 통과했다.
- **문자와 유효성의 혼동 가능성:** 공식 코드의 INVALID_STATE=2는 일반 상태 문자 D와 겹친다.
  첫·마지막 위치처럼 유효하지 않은 위치를 정상 검색 seed로 쓰지 않도록 8·10회차 통과 기준에 반영했다.
  첫날에는 일반 3Di 유효성 마스크를 구현했다고 주장하지 않는다.

### 오늘 이해할 핵심

PDB는 잔기 이름과 각 원자의 3차원 좌표를 담는다. CA는 잔기마다 있는 중심 탄소 원자다.
오늘 Python은 그 좌표를 읽어 표로 만들었고, 같은 PDB를 공식 Foldseek에 넣어 기준 문자열을 얻었다.
두 출력을 잔기별로 대응시킨 것이 오늘의 결과다. 좌표를 문자로 바꾸는 계산을 직접 작성하는 것이 다음 과제다.

설명 질문:

1. 첫 잔기의 PDB 번호가 1인데 sequence index는 0인 이유는?
2. 아미노산 문자 M과 공식 3Di 문자 D는 각각 무엇을 나타내는가?
3. CA 좌표가 있어도 3Di 계산이 무효가 될 수 있는 이유는?

### 다음 회차

`structures.py`를 N/CA/C/CB까지 확장한다. Gly에는 원래 CB가 없다는 사실과 구조 파일의
원자 결측을 구분하고, 좌표를 보정하기 전에 원본 존재 상태부터 출력한다.
가상 중심 계산과 자체 순전파는 각각 3·5회차에 진행한다.

## 이후 회차 기록 양식

- 회차 / 날짜 / 실제 학습 시간:
- 오늘의 개념을 자기 말로 설명:
- 구현·실행 명령과 산출물:
- 공식 비교의 분모 / 일치 / 불일치:
- 문제와 원인 / 해결 또는 미해결:
- 통과 기준 충족 여부:
- 다음 시작점:
