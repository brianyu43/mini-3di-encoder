# 기준 자료와 출처

확인일: 2026-09-15. 참고 소스는 references/upstream에 두고 자체 구현과 구분한다.
URL, 로컬 경로, SHA-256은 [source-manifest.json](references/source-manifest.json)에 기록한다.

| 자료 | 이 프로젝트에서 쓰는 용도 |
|---|---|
| [Foldseek 논문](https://doi.org/10.1038/s41587-023-01773-0) | 3Di 표현·진화적으로 보존되는 구조 상호작용이라는 연구 맥락. 이번 회차는 코드와 기준 출력 확인이며 논문 전체 재현 아님 |
| [Foldseek 고정 버전](https://github.com/steineggerlab/foldseek/tree/941cd33ff0771cd2e3f144e3293e22a2b87e9fda) | 실제 실행 기준. 버전 10-941cd33, 기존 로컬 공식 binary를 동일 hash로 복사 |
| [structureto3di.cpp](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/lib/3di/structureto3di.cpp) | 가상 중심, 상대 선택, 특징 10개, float 순전파 입력, 중심 선택, 마스크 |
| [structureto3di.h](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/lib/3di/structureto3di.h) | 20개 중심, 임베딩 2차원, 무효 상태 번호 2, 가상 중심 상수 |
| [공식 가중치](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/data/encoder_weights_3di.kerasify) | 5회차 자체 parser·순전파에 사용. 원본과 같은 1,032 bytes, 가중치·bias 242개 |
| [공식 치환 행렬](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/data/mat3di.out) | 공식 알파벳 검색 대조군. 재학습 알파벳에 무검증 재사용 금지 |
| [학습 코드 고정 snapshot](https://github.com/steineggerlab/foldseek-analysis/tree/654100b11242e581f9e6d43798b07a778903862e/training) | train_vqvae.py, 대응 특징 쌍 생성, 새 치환 행렬 추정 방법. Foldseek 실행 commit과 별개 snapshot이며 가중치 동일성은 추가 검증 필요 |
| [1UBQ / RCSB PDB](https://www.rcsb.org/structure/1UBQ) | 1회차의 작은 ubiquitin 구조. A 사슬 76잔기. 개발·설명용이고 최종 평가셋이 아님 |
| [1UBQ 원본 파일](https://files.rcsb.org/download/1UBQ.pdb) | 변형하지 않은 다운로드 원본. 잔기 번호·AA·CA 및 공식 출력의 공통 입력 |

## 확인한 중요한 차이

- 인코더의 `INVALID_STATE=2`는 일반 coil 상태와 같은 문자로 출력될 수 있다. 원본 3Di 문자열은 유효성 마스크를 대신하지 못한다.
- 첫날 표의 CA 좌표 존재는 3Di 특징 계산의 유효성 보장이 아니다. 공식 끝점 처리와 6잔기 주변 마스크는 추후 재현한다.
- 학습 소스는 `feat_x`로 예측한 평균·분산과 `feat_y`의 GaussianNLLLoss에 VQ 손실을 더한다.
  소규모 실험에서도 대응 구조의 관계를 목표로 하는지, 자기 복원인지 정확히 기록한다.
- training/README의 기본 스크립트는 데이터·정렬기 다운로드와 100개 seed 실험을 포함한다.
  이를 통째로 실행하지 않는다. 3주 차 CPU 소규모 예산을 따로 고정한다.
- 참고 Foldseek 소스의 라이선스는 upstream LICENSE.md에 보존되어 있다. 이번 실행 패키지에는
  공식 가중치·중심과 라이선스를 포함하고, 외부 raw C++ 소스·실행 binary는 Git 추적하지 않는다.

## 2–5회차 추가 확인

- [GemmiWrapper.cpp](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/src/strucclustutils/GemmiWrapper.cpp): 원본 좌표를 Vec3의 double 필드에 전달한다. 새 reader는 float64로 직접 읽는다.
- [Kerasify reader와 dense 계산](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/lib/kerasify/keras_model.cpp): little-endian uint32/float32 layout, 입력 차원 순서의 누산, bias와 ReLU/linear.
- [Util.cpp의 SSTR](https://github.com/steineggerlab/foldseek/blob/941cd33ff0771cd2e3f144e3293e22a2b87e9fda/lib/mmseqs/src/commons/Util.cpp): double 텍스트 출력은 `{:.3E}`. 내부 오차 비교와 출력 반올림 비교를 분리했다.
- 원본 소스는 수정하지 않았다. public/protected API로 중간값을 추적하는 별도 C++ harness를 만들었다.
  원본 전체 도구는 같은 배포 binary로 별도 비교했다. 상세 경계는 [THIRD_PARTY.md](THIRD_PARTY.md)에 기록했다.
