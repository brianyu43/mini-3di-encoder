# mini-3di-encoder

**단백질 좌표가 3Di 문자 한 글자로 바뀌는 과정을 직접 재구현하는 학습 프로젝트.**
공식 가중치로 인코더를 재현한 뒤, 작은 데이터로 재학습하고 기존 mini-3di-search에서 검색 성능을 비교한다.

- [3주·15회차 계획표](PLAN.md): 매일의 이론·구현·산출물·통과 기준
- [진행 기록](SESSION_LOG.md): 실제 완료한 것, 검증 결과, 다음 시작점
- [2–5회차 설명서](GUIDE_DAY02_05.md): 실제 잔기 하나로 계산을 따라가기
- [공식 구현과 데이터 출처](SOURCES.md)

## 지금 되는 것

**5회차까지 완료. 한 잔기의 좌표에서 3Di 문자까지 직접 계산하고 중간값을 추적한다.**
`atoms.py`가 원자 좌표를 읽고, `geometry.py`가 가상 중심·상대·특징 10개를 계산하며,
`network.py`가 공식 가중치로 순전파·중심 선택을 수행한다. `trace.py`가 과정을 JSON으로 보여 준다.
한 잔기를 추적하는 실행에는 공식 Foldseek나 PyTorch/TensorFlow 설치가 필요 없다.
가중치는 학습한 것이 아니라 공식 가중치를 재사용한다.

예: **1UBQ 잔기 2 → 상대 잔기 14 → 특징 10개 → (-2.476632,0.726601) → 상태 18 → W**.

1UBQ의 사전 선택한 위치 네 곳에서 원본 C++와 기하·특징·잠재 좌표·상태가 일치했다.
추가 신경망 입력 44개도 일치했다. 테스트 **42개 통과**, 별도 wheel 설치에서도 동일한 trace를 확인했다.
전체 사슬 검증(6회차), 일반 입력 처리·검색 연결(7–10회차), 재학습(11–15회차)은 아직 남아 있다.

| 결과 | 내용 |
|---|---|
| [2–5회차 결과 요약](results/day02-05/summary.json) | 단계별 native 비교와 검증 범위 |
| [원자 좌표 표](results/day02-05/day02_atoms.tsv) | N/CA/C/CB 좌표와 원본 존재 상태 |
| [한 잔기의 전체 계산](results/day02-05/trace_i001.json) | 후보 거리·특징·각 신경망 층·중심 거리·최종 문자 |
| [최종 설치·검증 기록](results/day05-validation/validation.json) | 테스트·Ruff·별도 wheel 설치 및 출력 동일성 |
| [원본 PDB](data/raw/1UBQ.pdb) | RCSB에서 받은 1UBQ 구조 |
| [좌표 표](results/day01/residues.tsv) | 잔기 번호, insertion code, AA, CA x/y/z, 공식 3Di 문자 |
| [공식 3Di](results/day01/official_3di.fasta) | Foldseek 10-941cd33이 출력한 76글자 |
| [공식 AA](results/day01/official_aa.fasta) | 같은 구조의 76잔기 아미노산 서열 |
| [실행 명령](results/day01/commands.json) | 실제 인자, return code, 시간 |
| [환경](results/day01/environment.json) | Python·패키지·OS·binary hash·프로세스 RSS |
| [검증](results/verification/validation.json) | 단위 검사, 원본 좌표 대조, 공식 출력 재실행 hash |

첫 잔기의 CA는 `(26.266, 25.413, 2.842)` Å이며 공식 문자는 D다.
끝점은 공식 인코더에서 무효 처리되지만 D로 나타날 수 있다. 따라서 **D라는 문자 자체가
유효·무효를 알려 주지는 않는다.** 이 표는 아직 검색용 유효성 마스크를 제공하지 않는다.

## 한 잔기를 직접 인코딩

프로젝트 폴더에서 실행한다. `.venv`는 로컬에 준비되어 있다.
출력은 아직 존재하지 않는 파일을 지정한다. index는 0부터 세므로 1은 두 번째 잔기다.

```bash
source .venv/bin/activate
python -m mini3di_encoder.trace \
  --structure data/raw/1UBQ.pdb --chain A --index 1 \
  --out artifacts/my-residue-2.json
```

이 명령은 우리 Python 구현과 패키지에 포함된 공식 가중치로 계산한다.
`day01.py`는 이전 단계의 공식 기준 출력 준비용으로 보존했다.

```bash
python -m pytest -q
ruff check .
ruff format --check .
python scripts/run_day02_05.py --out artifacts/my-day02-05-check
```

마지막 명령은 공식 C++ 소스를 컴파일해 선택한 잔기를 비교하고 원본 Foldseek의 descriptor와도 대조한다.
이 검증에는 현재 로컬의 참고 소스·Foldseek·clang++가 필요하다. 단위 테스트는 보존된 native 기준값을
사용하며, 스스로 기준 출력을 생성하거나 외부 Foldseek를 실행하지 않는다.

## 새 환경에서 설치

Python 3.12와 검증된 macOS arm64 환경을 기준으로 한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
```

공식 가중치·중심과 라이선스는 실행 패키지에 포함한다. Foldseek 실행 파일·참고 C++ 소스·가상 환경은
Git에 포함하지 않는다. 전체 검증 기준 자료를 복구할 때는
`references/source-manifest.json`의 URL/버전/hash를 따른다. 기존 Mac의 설치는 이전 프로젝트의
로컬 wheel을 이용해 오프라인으로 진행했으며 [wheel 기록](references/wheel-manifest.json)을 남겼다.
현재 `.venv`는 기존 프로젝트에 있던 Python 3.12 runtime을 기반으로 생성되었으므로 그 runtime을 삭제하면
가상 환경을 다시 만들어야 한다. 프로젝트 코드나 패키지 설치 내용은 별도다.

## 범위

일반적인 구조 파일의 모든 예외를 지원하지 않는다. 현재 원자 reader는 한 모델의 지정 사슬,
표준 ATOM 잔기를 다룬다. 대체 위치·비표준 잔기·TER 이후 다시 등장하는 사슬 등은 오류로 알린다.
빠진 원자는 삭제·0 대체하지 않고 존재 상태로 남기며, 해당 계산의 valid와 이유를 기록한다.
HETATM(물·리간드 등)은 제외한다. 모든 mmCIF·chain break·예외 입력을 공식과 동일하게 처리하는 단계는
아직 아니며 상세 정책은 8회차에서 확장한다. 외부 알고리즘·가중치의 출처는 [THIRD_PARTY.md](THIRD_PARTY.md)에 있다.
코드·계획은 AI 보조로 작성했으며 사용자의 실제 학습·이해 완료는 별도 확인한다.
