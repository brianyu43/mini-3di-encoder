# paired-vqvae-small-v1

실제 구조 대응 쌍으로 작게 재학습한 교육·연구용 인코더다. **공식 Foldseek 3Di와 다른 알파벳**이다.
기본 인코더의 공식 모델은 유지하며, 이 파일은 명시적으로 선택했을 때만 사용한다.

| 파일 | 의미 |
|---|---|
| `encoder.json` | BatchNorm을 접은 10→10→10→2 신경망 242개 파라미터와 20×2 중심 |
| `substitution.mat` | 이 상태 체계에 맞게 train에서 추정한 정수 점수 행렬; X=0 |
| `bundle.json` | 두 파일 hash, 선택 seed/epoch, validation에서 정한 gap open=14/extend=2 |

Python 3.12와 NumPy만 있으면 실행한다. 프로젝트 루트에서:

```bash
python -m mini3di_encoder.encode \
  --structure examples/1UBQ.pdb --chain A --record-id 1UBQ_A \
  --encoder-model models/paired-vqvae-small/encoder.json \
  --out artifacts/learned-example
```

query와 target 모두 이 모델로 인코딩하고 전용 행렬을 사용한다. 공식 알파벳의 기존 DB나
공식 행렬과 섞으면 상태 번호의 의미가 맞지 않는다. 두 모델의 같은 문자끼리 같은 기하 상태를
뜻하지 않는다. 무효 위치는 검색 mask와 X로 구분한다.

학습: SCOPe40 train 후보 256도메인 중 164개가 참여한 152개 구조 쌍, 양방향 특징 35,554개.
고정 기하 특징을 입력해 대응 특징을 예측하는 VQ-VAE다. 공개 Foldseek 학습 구조·손실을
재구현했고 새 파라미터·중심을 학습했다. seed 17/29/43 중 validation loss로 **29/epoch39**를
선택했다. 선택 데이터와 test는 fold/PDB/중복 hash가 겹치지 않는다.

작은 test 16query×48target에서 자체 검색 엔진 전수검색 MAP .9527, Recall@10 .9583이었다.
같은 엔진의 공식 표현+공식 행렬 MAP .9942보다 낮았다. 8 superfamily뿐이고 어려운 같은-fold
음성이 없으며 공식 공개 학습 목록과도 겹친다. Foldseek 대체 성능이나 새로운 구조 일반화는
검증하지 않았다. raw score는 확률이나 E-value가 아니다.

배포 NumPy 상태와 학습 checkpoint 상태는 57,642개 유효 특징에서 일치했다. 정확한 지표·
한계·단위는 [최종 보고서](../../REPORT_11_15.md), 실행 절차는 [재현 문서](../../REPRODUCE.md)에 있다.
코드·모델은 저장소의 [GPL 라이선스](../../LICENSE)와 출처 안내를 따른다.
