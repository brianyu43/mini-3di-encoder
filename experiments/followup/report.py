"""Render source-backed R1 result tables, PR plots and continuation evidence."""

import csv
import shutil

from experiments.common import ROOT, bounded_cpu, read, save, sha

from .common import OUT


def main():
    out = OUT
    bounded_cpu(out)
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = ROOT / "reports/followup-v2"
    destination.mkdir(parents=True, exist_ok=True)
    results = read(out / "R1-results.json")
    protocol = read(out / "protocol-v2.json")
    rows = results["rows"]
    assert read(out / "R1-independent-audit.json")["passed"]
    for name in [
        "R1-results.json",
        "R1-independent-audit.json",
        "data-freeze.json",
        "test-freeze.json",
        "protocol-v2.json",
        "manifest-v2.json",
        "denominators-v2.json",
        "selection-audit.json",
        "similarity-summary.json",
        "cost-preflight.json",
        "validation-selection.json",
        "encoding.json",
    ]:
        shutil.copyfile(out / name, destination / name)
    flat = []
    for row in rows:
        for setting in row["settings"]:
            for view, metrics in row["metrics"].items():
                flat.append(
                    {
                        "setting": setting,
                        "variant": row["variant"],
                        "gap_open": row["gap"][0],
                        "gap_extend": row["gap"][1],
                        "view": view,
                        **metrics["summary"],
                        "search_seconds": row["search_seconds_median"],
                    }
                )
    with (destination / "R1-results.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(flat)
    variants = protocol["variants"]
    labels = [
        "Official / original matrix",
        "Official / refitted matrix",
        "Learned / refitted matrix",
    ]
    colors = ["#315c91", "#bb7b22", "#28836c"]
    plt.rcParams.update(
        {"font.size": 10, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, setting in zip(axes, ["common", "selected"], strict=True):
        for index, (variant, color, label) in enumerate(zip(variants, colors, labels, strict=True)):
            row = next(r for r in rows if r["variant"] == variant and setting in r["settings"])
            x = np.arange(4) + (index - 1) * 0.18
            points = [row["metrics"][view]["summary"]["MAP"] for view in protocol["views"]]
            ci = [row["metrics"][view]["MAP_bootstrap"]["ci95"] for view in protocol["views"]]
            ax.errorbar(
                x,
                points,
                yerr=[
                    [p - interval[0] for p, interval in zip(points, ci, strict=True)],
                    [interval[1] - p for p, interval in zip(points, ci, strict=True)],
                ],
                fmt="o",
                color=color,
                label=label,
                capsize=3,
            )
        ax.set(
            xticks=range(4),
            xticklabels=["All", "Cross-family", "Same fold", "Other folds"],
            ylim=(0, 1.03),
            ylabel="MAP; 95% fold-cluster interval",
            title="Common gap (10, 1)" if setting == "common" else "Validation-selected gaps",
        )
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8, loc="lower left")
    for ext in ["png", "svg"]:
        fig.savefig(destination / f"R1-quality.{ext}", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, view in zip(axes, ["all", "cross_family"], strict=True):
        for variant, color, label in zip(variants, colors, labels, strict=True):
            m = next(r for r in rows if r["variant"] == variant and "common" in r["settings"])[
                "metrics"
            ][view]
            ax.plot(m["pr_recall"], m["pr_precision_macro_interpolated"], color=color, label=label)
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="Recall",
            ylabel="Macro interpolated precision",
            title="All targets" if view == "all" else "Same-family targets excluded",
        )
    axes[0].legend(fontsize=8)
    for ext in ["png", "svg"]:
        fig.savefig(destination / f"R1-precision-recall.{ext}", bbox_inches="tight")
    plt.close(fig)
    common = {r["variant"]: r for r in rows if "common" in r["settings"]}
    learned = common["learned_refit"]
    difference = (
        learned["metrics"]["all"]["summary"]["MAP"]
        - common["official_refit"]["metrics"]["all"]["summary"]["MAP"]
    )
    gate = {
        "R1_complete": True,
        "R2_MAP_gate_triggered": difference <= -0.02,
        "learned_minus_refit_common_gap_MAP": difference,
        "R2_failure_examples": len(
            [
                r
                for r in read(out / "R1-independent-audit.json")["poor_query_examples"]
                if r["variant"] == "learned_refit" and r["gap"] == [10, 1]
            ]
        ),
        "learned_score_time_fraction": learned["stage_seconds_median"]["sw_score"]
        / learned["search_seconds_median"],
        "R3_status": (
            "No filter-throughput conclusion from exhaustive R1 alone; "
            "preserve controlled-system plan"
        ),
        "script_sha256": sha(__file__),
    }
    save(out / "continuation-gates.json", gate)
    save(destination / "continuation-gates.json", gate)
    audit = read(out / "similarity-summary.json")
    counts = read(out / "selection-audit.json")
    preflight = read(out / "cost-preflight.json")
    table = []
    for setting in ["common", "selected"]:
        for variant in variants:
            row = next(r for r in rows if r["variant"] == variant and setting in r["settings"])
            values = row["metrics"]
            table.append(
                f"| {setting} | {variant} | {row['gap'][0]}/{row['gap'][1]} | "
                f"{values['all']['summary']['MAP']:.4f} | "
                f"{values['all']['summary']['recall_at_10']:.4f} | "
                f"{values['cross_family']['summary']['MAP']:.4f} | "
                f"{values['same_fold']['summary']['MAP']:.4f} | "
                f"{row['search_seconds_median']:.3f} |"
            )
    absent = counts["test_published_pdb_absent_queries"]
    report_text = f"""# R1: 어려운 음성을 포함한 고정 평가

이전 세 표현/행렬을 그대로 두고 평가 대상을 바꿨다. 결과를 본 뒤 구조나 설정을 바꾸지 않았다.
공식 모델/공식 행렬도 같은 mini-3di-search에서 평가했으며 Foldseek 실행 파일의 benchmark가 아니다.

## 데이터와 통과 기준

- Train은 이전 256도메인, paired 학습에 실제 참여한 구조는 164개다. 새 학습은 하지 않았다.
- 검색 validation은 이전 test fold b.34/c.26/d.17에서 30 query, 84 target을 구성했다.
  이전 test는 이제 개발 자료이며 독립 test라고 부르지 않는다. 이전 paired validation은
  모델 선택에 이미 사용됐으므로 새 test와의 노출 audit에도 포함했다.
- 새 test는 {results["query_count"]} query × {results["target_count"]} target,
  query SF {counts["test_query_sfs"]}개, fold {counts["test_folds"]}개다.
  모든 query에 같은-SF 양성, 같은-fold/다른-SF 음성, 다른-fold 음성을 확보했다.
  다른-family 양성도 최소 하나 있다. PDB·AA·구조 hash와 fold의 분할 겹침은 0이다.
- {audit["total_cross_pairs"]:,}개 cross-split/이전 노출 쌍 중 길이 상한상 가능한
  서열 {audit["sequence_pairs"]:,}쌍과 TM-align {audit["structure_pairs"]:,}쌍을 실제 비교했다.
  global identity ≥0.8 & 양쪽 coverage ≥0.9, min(TM1,TM2) ≥0.95의 근접 중복은 0이었다.
  이 기준보다 먼 유사성까지 없음을 보장하지 않는다. TM-align은 원본 알고리즘의 정렬을 사용한다.
- 첫 후보 목록에서 이전 자료와 같은 PDB인 항목을 발견해 **점수 계산 전에** 제외하고 다시
  선정했다. 첫 후보는 `artifacts/followup-preselection-v1`에 남겼다.

공식 공개 목록에 SID와 PDB가 모두 없는 query는 {len(absent)}개({", ".join(absent)})다.
이 둘의 지표는 JSON의 `published_pdb_absent` 층에 따로 있다. 목록 부재는 배포 가중치의
모든 학습 이력을 확인한 것이 아니므로, 이를 확실한 학습 비노출이나 일반화 검증이라고 부르지 않는다.
이 작은 층만으로 모델 간 결론을 내리지 않는다.

## 결과

공통 gap=(10,1) 결과와 validation에서 고른 gap 결과를 분리했다. MAP는 query 평균,
분모는 반환되지 않은 양성도 포함한다. `cross_family`는 같은-family target을 중립으로 제외한
다른-family 검색이다. `same_fold`는 같은 fold target만 남겨 어려운 음성에 집중한다.
동점은 score 내림차순/ID 오름차순, 동점 순서 기대 AP도 JSON에 보존했다.

| 설정 | 비교군 | gap | MAP 전체 | Recall@10 | MAP 다른 family | MAP 같은 fold | 검색 초 |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

![MAP와 불확실성](reports/followup-v2/R1-quality.png)

![고정 gap의 PR 곡선](reports/followup-v2/R1-precision-recall.png)

오차막대는 fold 전체를 2,000회 resample한 query 가중 평균의 95% percentile 구간이다.
fold마다 query 수가 달라도 원래 query 평균을 보존한다. 학습 seed 불확실성은 포함하지 않는다.
PR은 101개 recall 지점의 query별 보간 precision을 평균한 것이며, 그 면적을 MAP라고 하지 않는다.
공통 gap에서도 행렬은 각 알파벳에 맞는 서로 다른 행렬이므로 표현 단독의 인과 효과를 단정하지 않는다.

공통 gap에서 learned−official_refit MAP 차이는 **{difference:+.4f}**다. query별 실패 사례,
네 평가 층의 분모와 순위는 [독립 검산 결과](reports/followup-v2/R1-independent-audit.json)에 있다.
AP<0.5 사례는 원인 분석 후보이며 점수만으로 구조 표현의 실패 원인을 확정하지 않는다.

## 검증·비용

별도 코드가 실제 TSV를 다시 읽어 AP·Recall·대상 제외 규칙·fold bootstrap을 재계산했다.
세 반복의 점수/순위가 같고, top-1 경로는 Python 정렬·재채점으로 검사했다. 무효 mask/X를
포함한 인덱스 seed가 없는지도 확인했다. 시간은 warm Numba + top-1 Python traceback이며
인코딩·JIT·디스크 출력은 포함하지 않는다. 동일 gap 조건은 두 설정의 공통 결과로 표시했다.

validation 데이터로 {preflight["cells"]:,} DP cell을 계산하는 데 {preflight["seconds"]:.3f}초가
걸렸고, 사전 예측이 CPU 2시간/4GiB 한도 안이었다. 근접 중복 검사 wall time은
{audit["wall_seconds_this_invocation"]:.1f}초, 평가 프로세스 peak RSS는
{results["peak_process_rss_bytes_macos"] / 1e6:.1f}MB였다. 이 메모리는 부모 Python 프로세스의
누적 peak이며 컴퓨터 전체나 TM-align child와의 합산 최대치가 아니다. CPU 계산 스레드 1개,
GPU·클라우드·새 대규모 다운로드 없이 수행했다.

## 다음 작업과 재현

R2의 수치 gate(공통 gap MAP 차이 ≤−0.02)는 {gate["R2_MAP_gate_triggered"]}다.
R2 학습량·경계 안정성 실험은 R1 test를 보기 전에 `R2_PREREGISTRATION.json`에 조건을
기록했다. 진행 여부는 오류 사례와 이 gate를 함께 확인한다. R3의 필터 처리량은 R1 전수검색만으로
판정하지 않으며 별도 통제 실험이 필요하다. 목표는 아직 전체 후속 연구 진행 중이다.

코드: `experiments/followup/`. 원본 출력: `artifacts/followup-v2/`. 읽을 수 있는 결과:
`reports/followup-v2/`. 순서는 `prepare → similarity → freeze → encode → search validation
→ test 동결 커밋 → search test → audit_results → report`다. 이미 존재하는 출력은 덮어쓰지 않는다.
수치 설정·파일 hash는 [protocol-v2](reports/followup-v2/protocol-v2.json)와
[test freeze](reports/followup-v2/test-freeze.json)를 따른다. 원본 자료를 가진 이 환경에서는
각 모듈을 `.venv-research/bin/python -m experiments.followup.<module>`로 실행한다.
R1 전체를 새 위치에 재생성하려면 `common.OUT`을 별도 경로로 지정한 checkout에서 실행한다.
이 문서는 다른 컴퓨터에서의 전체 재현 검증을 주장하지 않는다.
"""
    (ROOT / "FOLLOWUP_R1_REPORT.md").write_text(report_text)
    print(gate, flush=True)


if __name__ == "__main__":
    main()
