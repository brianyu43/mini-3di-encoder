# mini-3di-encoder

**단백질 좌표가 3Di 문자 한 글자로 바뀌는 과정을 직접 재구현하는 학습 프로젝트.**
공식 가중치로 인코더를 재현한 뒤, 작은 데이터로 재학습하고 기존 mini-3di-search에서 검색 성능을 비교한다.

- [3주·15회차 계획표](PLAN.md): 매일의 이론·구현·산출물·통과 기준
- [진행 기록](SESSION_LOG.md): 실제 완료한 것, 검증 결과, 다음 시작점
- [공식 구현과 데이터 출처](SOURCES.md)

## 지금 되는 것

**1회차의 실행 산출물까지 완료. 자체 3Di 인코더는 아직 미구현.**
1UBQ A 사슬의 공식 Foldseek 출력과 76개 잔기의 번호·아미노산·Cα 좌표를 확보했다.
현재 `day01.py`는 공식 기준 출력을 준비하는 코드다. 좌표에서 특징·잠재 좌표·상태를 계산하는
자체 구현은 2–10회차에 추가한다. 재학습과 검색 비교는 11–15회차 계획이다.

| 결과 | 내용 |
|---|---|
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

## 현재 Mac에서 다시 실행

프로젝트 폴더에서 실행한다. `.venv`와 `tools/foldseek/bin/foldseek`는 로컬에 준비되어 있다.
출력은 아직 존재하지 않는 폴더를 지정한다.

```bash
source .venv/bin/activate
python -m mini3di_encoder.day01 \
  --structure data/raw/1UBQ.pdb --chain A \
  --foldseek tools/foldseek/bin/foldseek \
  --out artifacts/my-day01
```

Python 코드가 직접 처리하는 것은 현재 CA 좌표 표다. `createdb`와 `convert2fasta`로 확보한
3Di는 공식 출력이다. Foldseek v10의 헤더 공유는 AA/3Di/header의 DB key 일치를 검사한 후 적용한다.

```bash
python -m pytest -q
ruff check .
ruff format --check .
python scripts/verify_day01.py --out artifacts/my-day01-check
```

마지막 명령은 원본 출처 hash, PDB 고정 열과 좌표 표, 검사 도구, 공식 재실행을 확인한다.
공식 파일·실행 파일을 참조하므로 현재 로컬 자료가 필요하다. 단위 테스트 자체는 외부 Foldseek를 실행하지 않는다.

## 새 환경에서 설치

Python 3.12와 검증된 macOS arm64 환경을 기준으로 한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock.txt
python -m pip install --no-build-isolation --no-deps -e .
```

Foldseek 실행 파일·참고 소스·가상 환경은 Git에 포함하지 않는다. 전체 기준 자료를 복구할 때는
`references/source-manifest.json`의 URL/버전/hash를 따른다. 기존 Mac의 설치는 이전 프로젝트의
로컬 wheel을 이용해 오프라인으로 진행했으며 [wheel 기록](references/wheel-manifest.json)을 남겼다.
현재 `.venv`는 기존 프로젝트에 있던 Python 3.12 runtime을 기반으로 생성되었으므로 그 runtime을 삭제하면
가상 환경을 다시 만들어야 한다. 프로젝트 코드나 패키지 설치 내용은 별도다.

## 범위

일반적인 구조 파일의 모든 예외를 지원하지 않는다. 1회차 reader는 한 모델의 지정 사슬,
표준 ATOM 잔기, 유일하고 유한한 CA 좌표를 요구한다. 대체 위치·결측 CA·비표준 잔기는 오류로 알린다.
HETATM(물·리간드 등)은 제외한다. 상세 입력·유효성 규칙은 계획의 2·8회차에서 확장한다.
코드·계획은 AI 보조로 작성했으며 사용자의 실제 학습·이해 완료는 별도 확인한다.
