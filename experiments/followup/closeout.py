"""Write the final inventory only after every follow-up experiment audit succeeds."""

import subprocess

from experiments.common import ROOT, SEARCH_ROOT, environment, read, save

from .common import OUT


def main():
    r1 = read(OUT / "R1-results.json")
    r2 = read(OUT / "R2/decisions.json")
    r3 = read(OUT / "R3/system-results.json")
    audits = [
        read(OUT / path)
        for path in [
            "R1-independent-audit.json",
            "R2/final-audit.json",
            "R3/independent-audit.json",
        ]
    ]
    assert all(a["passed"] for a in audits)
    largest = next(c for c in r3["conditions"] if c["target_count"] == 5000)
    backend = largest["modes"]["double-ungapped"]["backends"]
    old = backend["baseline"]["top1"]["median"]
    new = backend["numba_traceback"]["top1"]["median"]
    rss = [r1["peak_process_rss_bytes_macos"], r3["peak_process_rss_bytes_macos"]]
    for name in ["training-results", "curve-results", "stability-results"]:
        rss.append(read(OUT / f"R2/{name}.json")["peak_process_rss_bytes_macos"])
    disk_kib = int(subprocess.check_output(["du", "-sk", str(OUT)], text=True).split()[0])
    source = subprocess.check_output(
        ["git", "-C", str(SEARCH_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    assert source == "4e8adf9d966dee34e581d13f4bb71f0db08a6ecd"
    assert not subprocess.check_output(
        ["git", "-C", str(SEARCH_ROOT), "status", "--porcelain"], text=True
    )
    save(ROOT / "reports/followup-v2/environment.json", environment())
    save(
        ROOT / "reports/followup-v2/completion.json",
        {
            "R1_complete": True,
            "R2_complete": True,
            "R3_complete_with_declared_data_shortfalls": True,
            "search_repository_unchanged_head": source,
            "largest_scale_backend_speedup": old / new,
            "largest_scale_filter_retention": largest["retain_exact_at_10"],
            "max_measured_python_peak_rss_bytes": max(rss),
            "local_artifact_disk_kib": disk_kib,
            "next_research_plan": "NEXT_RESEARCH_AFTER_FOLLOWUP.md",
            "future_study_executed": False,
            "gpu_used": False,
            "new_structure_downloads": False,
            "remote_push_performed": False,
        },
    )
    means = "/".join(f"{r['test_MAP_mean']:.4f}" for r in r2["fractions"])
    table = [
        "| R1: 어려운 평가 | 100×484, 56 SF/34 fold, 3비교군 | "
        "MAP .7502/.7486/.7412; 전체 우열 미확정 |",
        f"| R2: 학습·안정성 | 15학습, 2모델×6좌표 조건 | MAP 평균 {means}; soft seed 기준 미달 |",
        "| R3: CPU 구현 | 10조건×5backend×2모드×2출력×5반복 | "
        f"동일 결과, 5,000개 조건 {old / new:.2f}배 개선 |",
    ]
    report = f"""# 후속 연구 최종 보고

R1–R3의 실행 가능한 계획과 최종 검산을 마쳤다. 결과가 좋지 않은 학습 조건·soft seed·
표본 부족 조건도 그대로 남겼다. 다음 연구는 [별도 계획](NEXT_RESEARCH_AFTER_FOLLOWUP.md)이며
이번에 실행한 결과와 구분한다.

## 무엇을 확인했나

| 연구 | 실제 완료 범위 | 결론 |
|---|---|---|
{chr(10).join(table)}

R1의 learned−refit paired 95% 구간이 0을 포함해 전체 우열을 확정하지 않는다.
R2의 학습량 증가 효과는 seed마다 부호가 뒤집혔다. R3는 후보·점수·top-1 경로가 같았다.

R3의 5,000개 조건: 기존 구현 {old:.3f}초 → 최적화 구현 {new:.3f}초(10 query, warm,
top-1 상세 경로 포함, 5회 중앙값). 이때 필터의 전수검색 상위 10개 보존율은
{largest["retain_exact_at_10"]:.3f}이다. 구현 개선의 동등성과 필터가 만드는 손실은 별개의 결과다.
길이 구간별 5,000개는 실제 고유 구조가 부족해 미실행으로 명시하고, 혼합 길이에서만 실행했다.

## 결과를 읽는 순서

1. [R1: 데이터 분리·품질·오류 사례](FOLLOWUP_R1_REPORT.md)
2. [R2: 모든 seed·좌표 경계·실패한 soft seed](FOLLOWUP_R2_REPORT.md)
3. [R3: 동일 결과의 CPU 비용·전체 반복](FOLLOWUP_R3_REPORT.md)
4. [다음 연구의 20조건 요인 분리와 독립 평가 계획](NEXT_RESEARCH_AFTER_FOLLOWUP.md)

## 산출물과 검증

- `experiments/followup/`: 데이터 선정, 중복 audit, 동결, 학습, 검색, 좌표 변형,
  exact 후보/ungapped/traceback 커널, 독립 검산과 보고서 생성 코드.
- `models/learning-curve/`: 15개 인코더·대응 행렬·학습 선택 정보. 기존 공식 기본값과
  이전 배포 모델은 유지했다. 선택된 새 모델은 full-data seed71이며 연구용이다.
- `reports/followup-v2/`: 분모·manifest·freeze·모든 seed 지표·그림·검산·실행 소스 hash.
- `artifacts/followup-v2/`: 구조, 중간 특징, checkpoint, 점수 TSV, 전체 반복/실패 로그.
- R1 6,000개 query/view/run 검산, R2 점수표 78개 재검산, R3 timed run 1,000개 검산.
  정수 커널은 reference와 후보·점수·상세 경로가 같으며 경로 재채점도 통과했다.
- `mini-3di-search`의 추적 파일·HEAD는 바꾸지 않았다. 최적화는 이 저장소의 연구 커널이다.

## 필요한 자원과 한계

로컬 macOS arm64, Python 3.12, CPU 계산 스레드 1개로 실행했다. GPU/클라우드/새 구조
다운로드는 사용하지 않았다. 측정한 Python 프로세스 최대 RSS는 약 **{max(rss) / 1e6:.0f}MB**다.
운영체제·파일 cache·동시에 실행한 앱을 포함한 전체 RAM 사용량은 아니다. 후속 실험 원본과
중간 결과의 현재 디스크 사용량은 약 **{disk_kib / 1024 / 1024:.2f}GiB**이며 원래 보유하던
SCOPe 압축 파일과 이전 실험은 별도다. 단계별 시간과 포함/제외 비용은 각 보고서에 있다.

R1 공식 비교군도 자체 검색 엔진에서 평가했다. 공식 Foldseek 실행 파일과의 속도 비교가 아니다.
공식 모델의 공개 학습 목록 비중복 query가 2개뿐이므로 완전한 비노출 일반화 평가를 주장하지 않는다.
R2는 이미 본 R1 test에 대한 사전 등록 ablation이다. 작은 잡음은 실제 구조 오차의 물리 모델이 아니다.
이번 결과로 원 논문 전체 재현, 새로운 방법의 우월성, GPU 필요성을 주장하지 않는다.

실행 순서·원본 경로·해당 실험 코드 커밋은 [REPRODUCE](REPRODUCE.md)에 있다.
최종 코드 테스트와 lint 결과는 [실행 기록](FOLLOWUP_PLAN.md)에 기록한다.
"""
    (ROOT / "FOLLOWUP_REPORT.md").write_text(report)
    print("Follow-up closeout written; all three audit gates passed", flush=True)


if __name__ == "__main__":
    main()
