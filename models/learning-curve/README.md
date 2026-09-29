# R2 학습량 실험의 모든 모델

38/76/152 구조 쌍 각각에 seed 17/29/43/59/71을 적용한 15개 모델이다.
모든 실행의 optimizer step은 4,200이며 checkpoint는 paired validation loss로 선택했다.
primary는 전체 자료에서 고른 seed 71이다. 기존 공식 기본값이나 이전 배포 모델을 대체하지 않는다.

**각 encoder.json은 같은 폴더의 substitution.mat와 함께 사용한다.** 상태 번호는 공식 3Di나
다른 seed의 번호와 같은 의미가 아니다. query와 target을 모두 동일 모델로 다시 인코딩한다.
이번 비교의 gap은 open=10, extend=1이다. 모델 선택에 test를 사용하지 않았다.

| 폴더 | 선택 step | paired validation loss |
|---|---:|---:|
| [pairs-38-seed-17](pairs-38-seed-17/encoder.json) | 2500 | 0.169544 |
| [pairs-38-seed-29](pairs-38-seed-29/encoder.json) | 3800 | 0.172974 |
| [pairs-38-seed-43](pairs-38-seed-43/encoder.json) | 4100 | 0.171405 |
| [pairs-38-seed-59](pairs-38-seed-59/encoder.json) | 3400 | 0.179431 |
| [pairs-38-seed-71](pairs-38-seed-71/encoder.json) | 4100 | 0.165419 |
| [pairs-76-seed-17](pairs-76-seed-17/encoder.json) | 4200 | 0.185124 |
| [pairs-76-seed-29](pairs-76-seed-29/encoder.json) | 3400 | 0.173047 |
| [pairs-76-seed-43](pairs-76-seed-43/encoder.json) | 3900 | 0.196342 |
| [pairs-76-seed-59](pairs-76-seed-59/encoder.json) | 4000 | 0.171299 |
| [pairs-76-seed-71](pairs-76-seed-71/encoder.json) | 3800 | 0.175707 |
| [pairs-152-seed-17](pairs-152-seed-17/encoder.json) | 4200 | 0.180122 |
| [pairs-152-seed-29](pairs-152-seed-29/encoder.json) | 3400 | 0.176410 |
| [pairs-152-seed-43](pairs-152-seed-43/encoder.json) | 3100 | 0.198995 |
| [pairs-152-seed-59](pairs-152-seed-59/encoder.json) | 4200 | 0.172247 |
| [pairs-152-seed-71](pairs-152-seed-71/encoder.json) | 4000 | 0.169608 |

코드·해석·실패한 seed/수치 경계 사례는 [R2 보고서](../../FOLLOWUP_R2_REPORT.md),
학습 입력의 hash와 모든 점수는 [R2 결과](../../reports/followup-v2/R2/)를 따른다.
Torch checkpoint와 전체 학습 특징은 로컬 artifacts에 보존했다. 인코더 실행에는 NumPy만 필요하다.
