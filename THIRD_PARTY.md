# 외부 알고리즘·가중치·검증 코드

이 프로젝트의 3Di 기하 계산은 Foldseek의 공개 구현을 참고하여 Python으로 재구현했다.
신경망은 공식 고정 가중치를 사용하며 아직 학습하지 않았다.

- Foldseek: https://github.com/steineggerlab/foldseek/tree/941cd33ff0771cd2e3f144e3293e22a2b87e9fda
- 참고 함수: `lib/3di/structureto3di.cpp`의 기하·특징·상태 선택, `structureto3di.h`의 상수·중심.
- `src/mini3di_encoder/data/encoder.kerasify`: 원본 가중치의 byte-exact 사본, SHA-256은 `official.json`에 기록.
- `official.json`의 20개 중심과 기하 상수: 고정 원본 header에서 추출.
- Foldseek 라이선스 원문: 저장소 `LICENSE` 및 패키지 `data/FOLDSEEK-LICENSE.txt`에 보존.
- Kerasify의 원본 파일은 Robert W. Rose의 MIT 저작권 표시를 포함하며, 배포판이 포함한 LICENSE도
  `references/upstream/foldseek/lib/kerasify/LICENSE`에 보존했다.
- Python의 kerasify 파서는 같은 binary layout과 연산 순서를 독립적으로 구현했다.
  외부 `keras_model.cpp`를 런타임에 호출하지 않는다.

검증 때만 `validation/cpp_oracle.cpp`가 수정하지 않은 원본 `structureto3di.cpp`와
`keras_model.cpp`를 컴파일·호출한다. protected 함수는 상속으로 공개하며 private 접근이나
공식 알고리즘 수정은 하지 않는다. 빌드 때 원본 가중치 bytes를 C 배열 header로 변환한다.
원본 C++·SIMDe header·compiler·binary의 출처와 hash는 manifest 및 실험 실행 기록에 남긴다.
이 검증 harness는 자체 Python 인코더의 일부가 아니며 공식 전체 프로그램의 재빌드도 아니다.
