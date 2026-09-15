# mini-3di-encoder

PDB 원자 좌표에서 **한 잔기의 3Di 문자**를 계산하고 중간값을 출력하는 작은 인코더다.
가상 중심 → 상대 잔기 → 특징 10개 → 신경망 좌표 2개 → 20개 상태 중 선택을 직접 구현했다.
신경망 구조·가중치·중심은 공식 Foldseek 것을 사용하며 새로 학습하지 않았다.

## 설치와 실행

Python 3.12가 필요하다. 실행 의존성은 NumPy 하나다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m mini3di_encoder.trace \
  --structure examples/1UBQ.pdb --chain A --index 1 \
  --out artifacts/residue-2.json
```

`--index 1`은 사슬의 두 번째 잔기다. 출력 경로에는 새 파일을 지정한다.
예제 결과는 **Q(잔기 2) → 상대 잔기 14 → 상태 18 → W**다.
JSON에는 원자 존재 여부, 후보 거리, 특징, 신경망 각 층의 계산, 중심까지의 거리가 포함된다.
이 명령은 공식 Foldseek나 TensorFlow/PyTorch를 실행하지 않는다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `src/mini3di_encoder/atoms.py` | PDB 좌표·잔기 ID·빠진 원자 읽기 |
| `src/mini3di_encoder/geometry.py` | Cβ 근사, 가상 중심, 상대 선택, 특징 10개 |
| `src/mini3di_encoder/network.py` | kerasify 파일 읽기, 순전파, 최근접 중심 선택 |
| `src/mini3di_encoder/trace.py` | 위 계산을 연결하는 명령행 실행과 JSON 출력 |
| `src/mini3di_encoder/data/` | 공식 가중치·중심·라이선스 |
| `tests/` | 기하·신경망·입력 처리·독립 기준 출력 검사 |
| `examples/1UBQ.pdb` | 실행 예제와 테스트에 쓰는 원본 구조 |

## 검증과 범위

```bash
python -m pip install -r requirements-dev.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
ruff check .
ruff format --check .
```

Biopython은 독립 좌표 검증에만 사용한다. 테스트는 네 잔기의 원본 C++ 중간 계산과
별도 신경망 입력 44개를 보존된 기준 출력과 비교한다. 기준값은 테스트가 재생성하지 않는다.
원본 입력·기준 출력의 SHA-256과 생성 절차는 [fixture 출처](tests/fixtures/provenance.json)에 있다.

현재 검증은 1UBQ 한 구조의 선택한 유효 위치 4곳과 양 끝점에 한정된다.
한 모델의 지정 사슬·표준 ATOM 잔기를 지원하며 대체 위치, 비표준 잔기, TER 이후 사슬 재등장은 거부한다.
일반 mmCIF·사슬 단절 처리, 전체 사슬 대조, 검색 연결, 재학습은 아직 완료하지 않았다.
**문자 D는 유효 상태와 무효 위치 모두에서 나올 수 있으므로 반드시 `valid`를 함께 확인한다.**

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
  `main`은 인코더 실행과 회귀 검증에 필요한 파일만 유지한다.
