# mini-3di-encoder

PDB 원자 좌표를 **전체 사슬의 3Di 문자열**로 변환하고, 한 잔기의 중간 계산도 들여다보는 인코더다.
좌표 읽기 → 가상 중심 → 상대 잔기 → 특징 10개 → 신경망 좌표 2개 → 20개 상태 중 선택을 구현했다.
신경망 구조·가중치·중심은 공식 Foldseek 것을 사용하며 새로 학습하지 않았다.
단백질의 3차원 구조를 새로 예측하는 도구가 아니라, 주어진 구조를 검색용 문자로 바꾸는 도구다.

## 설치와 실행

Python 3.12가 필요하다. 실행 의존성은 NumPy 하나다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m mini3di_encoder.trace \
  --structure examples/1UBQ.pdb --chain A --index 1 \
  --out artifacts/residue-2.json
python -m mini3di_encoder.encode \
  --structure examples/1UBQ.pdb --chain A --record-id 1UBQ_A \
  --out artifacts/1ubq-chain
```

`--index 1`은 사슬의 두 번째 잔기다. 출력 경로에는 새 파일/폴더를 지정한다.
예제 결과는 **Q(잔기 2) → 상대 잔기 14 → 상태 18 → W**다.
JSON에는 원자 존재 여부, 후보 거리, 특징, 신경망 각 층의 계산, 중심까지의 거리가 포함된다.
이 명령은 공식 Foldseek나 TensorFlow/PyTorch를 실행하지 않는다.

전체 사슬 명령은 `chain.json`(위치별 상태·유효성·출처), `raw_3di.fasta`(원본 방식 문자),
`records.jsonl`(검색 입력)을 만든다. JSONL은 기존 `mini-3di-search`의 `index --real`과
`search --real`에 연결할 수 있다. 검색 엔진은 별도 저장소이며 인코더 실행 의존성이 아니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `src/mini3di_encoder/atoms.py` | PDB 좌표·잔기 ID·빠진 원자 읽기 |
| `src/mini3di_encoder/geometry.py` | Cβ 근사, 가상 중심, 상대 선택, 특징 10개 |
| `src/mini3di_encoder/network.py` | kerasify 파일 읽기, 순전파, 최근접 중심 선택 |
| `src/mini3di_encoder/trace.py` | 한 잔기의 계산 과정과 중간값 출력 |
| `src/mini3di_encoder/encode.py` | 전체 사슬 변환, 검색용 마스크와 JSONL 출력 |
| `src/mini3di_encoder/data/` | 공식 가중치·중심·라이선스 |
| `tests/` | 기하·신경망·입력 처리·독립 기준 출력 검사 |
| `examples/1UBQ.pdb` | 실행 예제와 테스트에 쓰는 원본 구조 |
| `validation/` | 고정한 13개 구조 목록, 공식 도구 대조·변환·검색 연결 검증 스크립트 |

## 입력과 유효성

- PDB의 지정 사슬을 읽는다. 모델이 여럿이면 `--model`, 대체 원자 위치가 있으면 `--altloc A`
  등으로 명시적으로 선택한다. 다른 모델이나 대체 위치의 좌표를 섞지 않는다.
- 표준 ATOM 잔기와 UNK를 지원한다. UNK의 아미노산은 X지만 좌표가 있으면 3Di를 계산한다.
  HETATM은 읽지 않으며, 다른 비표준 ATOM 잔기와 mmCIF는 명시적으로 거부한다.
- 빠진 N/Cα/C는 무효로 보존한다. 빠진 Cβ는 공식 기하 규칙으로 근사하되 관측 좌표와 구분한다.
  원본 잔기 번호·삽입 코드·좌표를 덮어쓰지 않는다. Cα 결측 잔기도 위치를 보존하는 정책은
  이를 생략할 수 있는 공식 파일 파서와 다르므로, 그런 입력의 문자열을 그대로 대조하면 안 된다.
- `feature_valid_mask`는 공식 방식의 기하·주변 잔기 유효성이다. `trace`의 `valid`도 이 의미다.
  `valid_seed_mask`는 여기에 사슬 연결 검사를 추가한다. TER 경계 또는 C(i)–N(i+1) 거리의
  결측/0/2Å 초과를 단절로 보고, 자기와 상대의 주변이 단절된 위치를 검색 seed에서 제외한다.
  2Å는 이 프로젝트의 보수적인 검색 정책이며 공식 학습 파라미터가 아니다. 번호의 도약만으로
  사슬을 끊지는 않는다. `residue_valid_mask`는 각 잔기의 기하 준비 단계 유효성이다.
- 무효 검색 위치는 길이를 유지한 채 X로 출력한다. **D는 유효 상태와 무효 위치 모두에서
  나올 수 있으므로 문자만으로 판정하지 않는다.** 검색 엔진은 마스크가 거짓이거나 X를 포함한
  seed를 제외한다. 공식 행렬에서 X 점수는 0이므로 정렬 경로 자체는 X를 지나갈 수 있다.

## 검증과 범위

```bash
python -m pip install -r requirements-dev.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
ruff check .
ruff format --check .
```

Biopython은 독립 좌표 검증에만 사용한다. 테스트는 전체 1UBQ 사슬의 원본 C++ 중간 계산과
공식 배포 도구의 문자열, 별도 신경망 입력 44개, 입력 경계 처리를 검사한다.
기준값은 테스트가 재생성하지 않는다.
원본 입력·기준 출력의 SHA-256과 생성 절차는 [fixture 출처](tests/fixtures/provenance.json)에 있다.

6–10회차 개발 검증에서는 1UBQ와 기존 SCOPe 개발 구조 12개에서 **1,384/1,384 위치의
공식 문자가 일치**했다. 이 중 공동 유효 위치는 1,358/1,358이며 특징·잠재 좌표도 일치했다.
39회 강체 변환에서 상대·상태·검색 마스크가 유지됐다. PDB 소수 세 자리 반올림 후 두 위치가
달라졌지만, 같은 반올림 입력에서는 공식 도구도 동일한 상태를 출력했다.
짧은 사슬과 원자 결측 7건은 좌표 수준에서 원본 C++과 추가 대조했다.

검색 연결은 6×6 개발 통합 시험이다. 전수/단일 seed는 36쌍, 두 seed/ungapped 추가는 21쌍을
정렬했고, 유지된 쌍의 점수는 전수검색과 같았다. 실제 seed 1,868개 중 무효 위치를 포함한 것은
0개였다. 이는 구현 일치와 연결 검증이며 독립적인 생물학적 정확도·일반화 성능의 증거는 아니다.
자세한 분모·실행 비용·재현 명령은 [6–10회차 계획과 결과](PLAN_06_10.md)에 있다.

## 출처와 연구 기록

- 기준 구현: [Foldseek 10-941cd33](https://github.com/steineggerlab/foldseek/tree/941cd33ff0771cd2e3f144e3293e22a2b87e9fda).
  기하·특징·중심 선택은 `lib/3di/structureto3di.cpp`와 `.h`를 참고해 재구현했다.
- [공식 가중치](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/data/encoder_weights_3di.kerasify)는 원본과 같은 1,032 bytes다.
  SHA-256과 중심은 [official.json](src/mini3di_encoder/data/official.json)에 보존했다.
- Kerasify는 Robert W. Rose의 MIT 라이선스 도구다. 파일 형식과 연산 순서를 Python으로 구현했으며
  외부 C++ 코드는 실행 패키지에 포함하지 않는다. Foldseek 코드·가중치의 GPL 원문은 [LICENSE](LICENSE)에 있다.
- 예제 구조: [RCSB 1UBQ](https://www.rcsb.org/structure/1UBQ), 원본 파일을 수정하지 않았다.
- **1–5회차 계획·설명서·전체 실험 기록·공식 검증 스크립트**는
  [research/sessions-01-05](https://github.com/brianyu43/mini-3di-encoder/tree/research/sessions-01-05)에 보존했다.
  큰 입력·중간 DB·실험 로그는 Git에 넣지 않고 `artifacts/`에 둔다.
