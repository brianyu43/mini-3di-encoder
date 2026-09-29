# 11–15회차 재현

Python 3.12, CPU, C++ 컴파일러가 필요하다. 아래 명령은 이 저장소 루트에서 실행한다.
기존 결과를 덮어쓰지 않도록 `artifacts/reproduction`은 새 경로여야 한다.
학습은 선택 사항이며, 배포 모델로 구조를 변환하는 데 PyTorch는 필요 없다.

## 저장소만으로 실행

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m mini3di_encoder.encode \
  --structure examples/1UBQ.pdb --chain A --record-id 1UBQ_A \
  --encoder-model models/paired-vqvae-small/encoder.json \
  --out artifacts/learned-1ubq
```

공식 모델을 쓰려면 `--encoder-model`을 생략한다. 두 모델의 문자열은 같은 알파벳으로
해석하면 안 된다. 재학습 문자열에는 `models/paired-vqvae-small/substitution.mat`를 쓴다.
기존 공식 3Di DB를 재사용하지 말고 query와 target 모두 같은 모델로 다시 인코딩한다.

## 학습·검색 전체 재현

```bash
python3.12 -m venv .venv-research
.venv-research/bin/python -m pip install -r requirements-research.lock.txt
.venv-research/bin/python -m pip install --no-build-isolation --no-deps -e .
```

의존성 lock은 이번 macOS arm64/Python 3.12 환경이다. 다른 운영체제에서의 수치·시간 일치까지
검증한 것은 아니다. `mini-3di-search`는 기본적으로 이 저장소와 나란히 있어야 한다.
다른 위치라면 `MINI3DI_SEARCH_ROOT`에 그 저장소 루트를 지정한다. 원 실험의 검색 코드
SHA-256 목록은 `reports/sessions-11-15/search-results.json`에 있다. 해당 저장소를 수정하지 않는다.

원본 [SCOPe40 구조 묶음](https://wwwuser.gwdguser.de/~compbiol/foldseek/scop40pdb.tar.gz)은
306,064,157 bytes, SHA-256 `8bac002ff3c1329beaf14d6a645fab249b6dd2aae98e2093b40a6f49d3a60fa4`다.
기존 로컬 파일을 사용한다. 이 저장소에는 PDB 묶음을 넣지 않았으며 재현 스크립트가 이를
자동 다운로드하지도 않는다. 구조를 입수할 때 원본 데이터의 출처와 이용 조건을 보존한다.

```bash
.venv-research/bin/python -m experiments.restore \
  --archive /path/to/scop40pdb.tar.gz --out artifacts/reproduction \
  --allow-small-downloads
.venv-research/bin/python -m experiments.features --out artifacts/reproduction
.venv-research/bin/python -m experiments.train --out artifacts/reproduction \
  --upstream-training-source artifacts/reproduction/references/train_vqvae.py
.venv-research/bin/python -m experiments.matrices --out artifacts/reproduction \
  --official-matrix artifacts/reproduction/references/mat3di.out
.venv-research/bin/python -m experiments.evaluate validation --out artifacts/reproduction
.venv-research/bin/python -m experiments.evaluate test --out artifacts/reproduction
.venv-research/bin/python -m experiments.audit --out artifacts/reproduction
.venv-research/bin/python -m experiments.benchmark_encoding --out artifacts/reproduction
```

`restore`는 고정 manifest의 384개 구조만 직접 읽어 쓰고 hash를 대조한다. tar의 경로를 그대로
풀지 않는다. 기존 결과 폴더의 소스 캐시가 있으면 재사용한다. 허용 플래그가 있을 때만
고정된 TM-align 소스·공식 행렬·학습 코드/목록 총 500,539 bytes를 받을 수 있다. 모두 hash를
검사하고 컴파일한다. 다운로드 없이 실행하려면 `--source-cache`로 이 파일들이 있는 실험
폴더를 지정하고 허용 플래그를 생략한다. 중간 실패 후에는 원인 로그를 보존하고 새 출력
폴더를 사용한다. TM-align 바이너리 hash는 컴파일러에 따라 달라질 수 있다.

`experiments/frozen/`은 원 실험의 **데이터/평가 동결 증거**다. 그 안의 `test-freeze.json`의
`test_search_has_not_run: true`는 동결 당시의 상태다. 완료 후에도 원본을 수정하지 않았다.
재현 실행은 validation 종료 시 자기 출력·환경에 맞는 새 freeze를 만든다. 원 실험의 hash를
새 실행에 억지로 복사하지 않는다. `data-freeze`의 catalog/plan hash는 원 선택 기록에 대한
참조이며, 복원 경로는 후보 선정을 다시 하지 않는다. 원 전체 catalog는 로컬 artifacts에 있다.

원 실험과의 숫자 비교:

```bash
.venv-research/bin/python -m experiments.verify_replay \
  --original artifacts/sessions-11-15 --replay artifacts/reproduction
```

이 명령은 구조별 특징 배열, 학습 쌍, 480개 정렬 선택, 세 모델의 파라미터·중심,
183개 epoch 지표와 세 행렬을 대조한다. wall time과 JSON의 실행 소스 경로 정보는 비교하지 않는다.
이번에 `artifacts/sessions-11-15-replay`에서 이 전체 재생성 검사가 실제 통과했다.

## 검증·그림

```bash
MINI3DI_TRAINING_REFERENCE=artifacts/reproduction/references/train_vqvae.py \
  .venv-research/bin/python -m pytest -q
.venv-research/bin/ruff check .
.venv-research/bin/ruff format --check .
```

고정 공식 학습 소스가 없으면 그 소스와의 순전파·gradient 대조 1개만 skip한다. PyTorch가
없는 기본 환경에서는 학습 전용 테스트 모듈을 skip하고 인코더·행렬·지표 검사를 실행한다.
학습 소스는 hash 검사 후 테스트에서만 import한다. 훈련기는 자체 구현을 사용한다.

`python -m experiments.report`는 원 결과 위치 `artifacts/sessions-11-15`를 읽어
`reports/sessions-11-15`의 JSON/TSV/PNG/SVG와 배포 모델 bundle을 갱신한다.
다른 재현 결과를 원 결과 위에 덮어써 보고하지 않는다. 새 결과 보고서는 별도 경로로 작성한다.

## 검색 CLI로 연결

아래는 재현 실행이 만든 learned query/target JSONL을 쓰는 예다. `PYTHONPATH`는
read-only 검색 소스를 읽기 위한 것이며 그 저장소에 설치·수정을 하지 않는다.

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="../mini-3di-search/src"
export NUMBA_CACHE_DIR="$PWD/artifacts/reproduction/numba-cache"
.venv-research/bin/python -m mini3di_search.cli index --real --k 3 \
  --records artifacts/reproduction/search-test/learned-target.jsonl \
  --out artifacts/reproduction/learned-cli-index.json
.venv-research/bin/python -m mini3di_search.cli search --real \
  --queries artifacts/reproduction/search-test/learned-query.jsonl \
  --db artifacts/reproduction/learned-cli-index.json \
  --matrix artifacts/reproduction/matrices/learned_refit.mat \
  --matrix-source paired-vqvae-small --gap-open 14 --gap-extend 2 \
  --mode exhaustive --top-k 1 --out artifacts/reproduction/learned-cli-search
```

API 실험과 같은 top-1 상세 경로·모든 양수 점수 출력 조건이다. 전체 검색 결과에는
E-value·TM-score·상동성 확률을 붙이지 않는다.

## 후속 R1–R3 실험

동일한 research lock 환경을 사용한다. 후속 실행은 기존 원 실험의 catalog, 152개 train
구조 쌍, 모델, 공식 목록과 로컬 구조 archive를 재사용한다. 해당 파일은 Git에 모두 포함되어
있지 않으므로 **새 clone에서 아래 명령만 실행하면 데이터까지 자동 복구되는 것은 아니다**.
`experiments/followup/common.py`의 ORIGINAL/OUT, `experiments.prepare.ARCHIVE` 경로를
확인한다. 다른 머신에서 전체 재현을 별도로 검증한 것은 아니다.

실제 실행한 단계:

```bash
.venv-research/bin/python -m experiments.followup.prepare
.venv-research/bin/python -m experiments.followup.similarity
.venv-research/bin/python -m experiments.followup.freeze
.venv-research/bin/python -m experiments.followup.encode
.venv-research/bin/python -m experiments.followup.search validation
# 이 지점에서 test-freeze와 선택 설정을 커밋하고, 그 뒤 test를 실행한다.
.venv-research/bin/python -m experiments.followup.search test
.venv-research/bin/python -m experiments.followup.audit_results
.venv-research/bin/python -m experiments.followup.report
.venv-research/bin/python -m experiments.followup.train_curve
.venv-research/bin/python -m experiments.followup.evaluate_curve
.venv-research/bin/python -m experiments.followup.stability
.venv-research/bin/python -m experiments.followup.final_audit
.venv-research/bin/python -m experiments.followup.report_r2
.venv-research/bin/python -m experiments.followup.system_prepare
.venv-research/bin/python -m experiments.followup.system_benchmark
.venv-research/bin/python -m experiments.followup.report_system
```

완료 결과가 존재하는 현재 OUT에 그대로 재실행하지 않는다. 별도 checkout에서 OUT을
새 경로로 지정하고 각 단계의 freeze를 새로 만든다. 원본 R1 test-freeze는 첫 실행의
hash 함수 이름 충돌 수정 전 코드 hash를 포함하며, `execution-repair.json`으로 수정 이력을
연결했다. 수정 전 코드는 `cdbb22e` 이전 커밋, 실제 수정 실행은 `cdbb22e`에 보존했다.
현재 실행 스크립트의 편의 수정까지 원래 실험 코드였다고 간주하지 않는다.

R2 평가 실행 소스는 `d22a2a7`의 evaluate_curve 및 그 직전 `4f6e9a2`의 evaluation이다.
변형 좌표 실험은 `d22a2a7`, R3 준비는 `85c7402`, R3 측정은 `898467f`의 실행 코드다.
각 artifacts의 freeze에 실제 파일 hash도 보존했다. 생성 보고서·검산 스크립트는 뒤에 추가했다.

R2의 15개 NumPy 배포 모델과 전용 행렬은 `models/learning-curve/`에서 바로 읽을 수 있다.
모두 연구 결과이며 기존 공식 기본값/배포 bundle을 대체하지 않았다. 모델 사용 시 대응하는
행렬을 함께 사용하고, 기존 3Di DB도 같은 모델로 다시 인코딩한다.

Numba 또는 나란한 검색 저장소가 없는 기본 환경에서는 두 후속 시스템/soft-seed 테스트
모듈도 skip한다. `python -m pytest`로 실행하면 core 인코더 검증은 계속 동작한다.
