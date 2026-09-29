"""Report every learning run, numerical boundary case and failed soft-seed gate."""

import csv
import shutil

import numpy as np

from experiments.common import ROOT, bounded_cpu, read, save

from .common import OUT


def main():
    out = OUT / "R2"
    bounded_cpu(out)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training = read(out / "training-results.json")
    curve = read(out / "curve-results.json")
    stability = read(out / "stability-results.json")
    audit = read(out / "final-audit.json")
    assert audit["passed"]
    dest = ROOT / "reports/followup-v2/R2"
    dest.mkdir(exist_ok=True, parents=True)
    for name in [
        "protocol.json",
        "data-freeze.json",
        "training-results.json",
        "curve-results.json",
        "search-freeze.json",
        "stability-results.json",
        "final-audit.json",
    ]:
        shutil.copyfile(out / name, dest / name)
    for run in training["runs"]:
        name = f"pairs-{run['structure_pairs']}-seed-{run['seed']}"
        target = ROOT / "models/learning-curve" / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ["encoder.json", "substitution.mat", "summary.json"]:
            shutil.copyfile(out / name / file, target / file)
        shutil.copyfile(out / name / "history.json", dest / f"{name}-history.json")
    table, summaries, raw = [], [], []
    for count in [38, 76, 152]:
        runs = [r for r in curve["runs"] if r["structure_pairs"] == count]
        vals = [r["splits"]["validation"]["metrics"]["all"]["summary"]["MAP"] for r in runs]
        tests = [r["splits"]["test"]["metrics"]["all"]["summary"]["MAP"] for r in runs]
        summary = {
            "structure_pairs": count,
            "test_MAP_mean": float(np.mean(tests)),
            "test_MAP_sample_sd": float(np.std(tests, ddof=1)),
            "validation_MAP_mean": float(np.mean(vals)),
            "seeds": len(runs),
        }
        summaries.append(summary)
        table.append(
            f"| {count} ({count / 152:.0%}) | {np.mean(vals):.4f} | {np.mean(tests):.4f} | "
            f"{np.std(tests, ddof=1):.4f} |"
        )
        for r in runs:
            raw.append(
                {
                    "pairs": count,
                    "seed": r["seed"],
                    "validation_loss": r["validation_loss"],
                    "validation_MAP": r["splits"]["validation"]["metrics"]["all"]["summary"]["MAP"],
                    "test_MAP": r["splits"]["test"]["metrics"]["all"]["summary"]["MAP"],
                }
            )
    with (dest / "learning-curve.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(raw)
    delta = []
    for seed in [17, 29, 43, 59, 71]:
        by_count = {r["pairs"]: r["test_MAP"] for r in raw if r["seed"] == seed}
        delta.append({"seed": seed, "full_minus_quarter_MAP": by_count[152] - by_count[38]})
    decision = {
        "fractions": summaries,
        "paired_seed_deltas": delta,
        "all_seed_signs_positive": all(d["full_minus_quarter_MAP"] > 0 for d in delta),
        "all_seed_signs_negative": all(d["full_minus_quarter_MAP"] < 0 for d in delta),
        "soft_seed_adopted": any(s["passed"] for s in stability["selection"].values()),
    }
    save(dest / "decisions.json", decision)
    save(out / "decisions.json", decision)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for seed in [17, 29, 43, 59, 71]:
        values = [
            next(r["test_MAP"] for r in raw if r["pairs"] == c and r["seed"] == seed)
            for c in [38, 76, 152]
        ]
        axes[0].plot([25, 50, 100], values, "o-", alpha=0.65, label=f"Seed {seed}")
    axes[0].set(
        xlabel="Training structure pairs (%)",
        ylabel="Test MAP",
        ylim=(0.65, 0.78),
        title="Equal 4,200 optimizer steps; all seeds",
    )
    axes[0].legend(fontsize=8)
    for model, color in [("official", "#315c91"), ("learned", "#28836c")]:
        runs = [r for r in stability["runs"] if r["model"] == model and r["split"] == "test"]
        axes[1].plot(
            range(len(runs)),
            [r["counts"]["state_changes"] / r["counts"]["common_valid"] for r in runs],
            "o-",
            color=color,
            label=model,
        )
    axes[1].set(
        xticks=range(6),
        xticklabels=["Original", "Rigid", "Round", ".001 Å", ".01 Å", ".05 Å"],
        ylabel="Changed-state fraction (joint valid positions)",
        title="Coordinate stress",
    )
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    for ext in ["png", "svg"]:
        fig.savefig(dest / f"learning-stability.{ext}", dpi=160)
    plt.close(fig)
    stable_rows, soft_rows = [], []
    for r in stability["runs"]:
        if r["split"] != "test":
            continue
        c = r["counts"]
        stable_rows.append(
            f"| {r['model']} | {r['condition']} | "
            f"{c['partner_changes'] / c['common_valid']:.2%} | "
            f"{c['state_changes_same_partner'] / c['same_partner_positions']:.2%} | "
            f"{c['broken_3mers'] / c['valid_3mers']:.2%} | "
            f"{r['exhaustive']['metrics']['all']['summary']['MAP']:.4f} | "
            f"{r['filters']['double']['metrics']['all']['summary']['MAP']:.4f} | "
            f"{r['filters']['double']['retain_exact_at_10']:.3f} |"
        )
    for name, selection in stability["selection"].items():
        for t in selection["trials"]:
            soft_rows.append(
                f"| {name} | {t['threshold']} | {t['retain_exact_at_10']:.3f} | "
                f"{t['candidate_pairs']} | {t['candidate_seconds']:.3f} |"
            )
    primary = next(r for r in raw if r["pairs"] == 152 and r["seed"] == curve["primary_seed"])
    resource = [training, curve, stability]
    delta_text = ", ".join(f"{d['seed']}: {d['full_minus_quarter_MAP']:+.4f}" for d in delta)
    margin_rows = []
    for r in stability["runs"]:
        if r["condition"] != "gaussian_0.01_A" or r["split"] != "test":
            continue
        c = r["counts"]
        for label, key in [("0–0.01", "0_0.01"), ("0.25–1", "0.25_1.000001")]:
            total, changed = c[f"margin_{key}_positions"], c[f"margin_{key}_changes"]
            margin_rows.append(
                f"| {r['model']} | {label} | {changed}/{total} | {changed / total:.2%} |"
            )
    report = f"""# R2: 학습량과 상태 경계 실험

사전에 정한 15회 학습·검색과 6가지 좌표 조건을 모두 실행했다. 학습량 증가의 효과가
seed마다 같은 방향으로 나타나지 않았으므로 **학습 데이터를 늘리면 개선된다는 결론을 보류한다**.
Soft seed도 validation 보존율 기준을 통과하지 못했다. 기존 배포 모델은 교체하지 않았다.

## 학습량: 모든 초기값을 공개

같은 31 superfamily를 포함하는 중첩 구조 쌍 38/76/152개에서 각각 seed 5개를 학습했다.
실제 directed residue-pair 수는 [data-freeze](reports/followup-v2/R2/data-freeze.json)에 있다.
모든 실행은 Adam 0.001, batch 512, 4,200 step이다. epoch 수가 같은 비교가 아니다.
checkpoint는 원래 paired validation loss의 최솟값으로 선택했다. 새로운 학습 쌍 정렬은
필요하지 않아 기존 152쌍과 정렬 audit를 재사용했다. 학습·모델 선택에 R1 test를 쓰지 않았다.

| 구조 쌍 | validation MAP 평균 | test MAP 평균 | seed 표본 표준편차 |
|---|---:|---:|---:|
{chr(10).join(table)}

각 seed의 full−quarter MAP 차이: {delta_text}.
seed 평균·표준편차는 이 5개 초기값의 기술 통계이며 데이터셋 불확실성의 신뢰구간이 아니다.
표현과 **치환 행렬도 같은 비율의 학습 쌍**에서 얻었으므로 이 표로 표현 학습량만의 효과를
분리할 수 없다. 다음 실험에서 두 요인을 분리해야 한다.

primary는 full-data 중 paired loss로 고른 seed **{curve["primary_seed"]}**,
validation MAP {primary["validation_MAP"]:.4f}, test MAP {primary["test_MAP"]:.4f}다.
validation 검색 MAP로 고르면 seed {curve["alternative_validation_MAP_seed"]}가 된다.
이 대안 규칙의 비교는 validation에서만 해석하고, test에 맞춰 primary를 바꾸지 않았다.
전체 seed 수치: [TSV](reports/followup-v2/R2/learning-curve.tsv).

![학습량과 좌표 안정성](reports/followup-v2/R2/learning-stability.png)

## 좌표 변화: 두 가지 불안정성을 구분

질의와 대상 모두에 같은 조건을 적용했다. 강체변환은 한 회전/이동, Gaussian은 구조별로
고정된 독립 원자 잡음이며 강도별로 같은 표준정규 표본을 배율만 바꿨다. 실제 구조 오차의
확률모형이 아니다. 원본 결측 원자는 결측으로 남기고 동일한 기하 규칙으로 처리했다.

| 모델 | 조건 | 상대 변경 | 상대 고정 시 상태 변경 | 3-mer 파괴 | 전수 MAP | 필터 MAP | retain@10 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stable_rows)}

상대·상태 변경률은 두 조건 모두 검색에 유효한 위치를 분모로 한다. mask 변경은 별도 JSON에
보존했다. 3-mer는 원본에서 유효했던 창을 분모로, 상태 변경 또는 새 무효 위치가 생기면
파괴로 센다. 두 모델은 강체변환 후 상태·mask·상대가 모두 같았다. 반올림과 잡음에서는
경계 통과가 생겼다. 중심 거리 margin별·구조별 변경 수와 분모도 저장했다.
MAP 변화는 단조롭지 않으며 이 표에서 특정 잡음이 성능을 개선한다고 일반화하지 않는다.

0.01Å 조건에서 상대 잔기가 유지된 위치만 보면, 원본 중심 거리 margin이 작은 위치에서
상태 변경이 훨씬 잦았다. 아래는 미리 정한 구간 중 가장 낮은/높은 구간이다.
중간 구간 역시 JSON에 전부 있다.

| 모델 | 원본 normalized margin | 상태 변경/위치 수 | 변경률 |
|---|---|---:|---:|
{chr(10).join(margin_rows)}

margin이 작은 위치는 두 중심 중 어느 쪽을 고를지 경계에 가깝다. 다만 이것을 표시하는 것과
검색 후보를 충분히 복구하는 것은 별개의 문제였고, 아래 soft seed는 기준을 통과하지 못했다.

## Soft seed: 실패도 결과

유효한 query 3-mer에서 한 위치에만 두 번째 중심을 허용하고 positional hit는 중복 제거했다.
변경은 후보 검색에만 적용하며 정렬 점수와 문자는 바꾸지 않는다. 기준은 원본 validation의
retain_exact@10 ≥0.98, 그 안에서 후보 수 최소였다.

| 모델 | margin 상한 | retain@10 | 후보 쌍 | 후보 생성 초 |
|---|---:|---:|---:|---:|
{chr(10).join(soft_rows)}

두 모델 모두 기준 미달이므로 soft seed의 test 설정을 새로 고르지 않았다. 위 좌표 표는
원래 double seed를 쓴다. 필터 MAP는 전수 점수표에서 해당 후보만 남겨 재순위화했다.
필터의 후보 생성 시간은 실측했지만, 이 R2 표는 필터 SW까지 다시 실행한 속도 benchmark가 아니다.
속도 비교는 별도 R3에서 한다.

## 검증·비용·한계

- 저장된 **{len(audit["persisted_tables_checked"])}개 점수 TSV**를 다시 읽어 4개 평가 층의
  AP·Recall·분모·순위를 독립 계산했다. 공식 batch encoder의 원본 조건 결과도 R1과 같았다.
- 15모델 중 76쌍/seed17의 학습 입력 한 곳에서 float32 BN folding 후 상태가 달라졌다.
  약 2.5e−7의 잠재 좌표 변화가 중심 경계를 넘은 사례이며 상세 거리 차이는
  [검증 기록](reports/followup-v2/R2/final-audit.json)에 있다. 행렬과 검색은 모두 배포된
  NumPy 상태를 사용해 일관성을 유지한다. 모든 상태가 Torch와 같다고 주장하지 않는다.
- 학습 {training["wall_seconds"]:.1f}초, 15모델 검색·검산 {curve["wall_seconds"]:.1f}초,
  좌표 실험 {stability["wall_seconds"]:.1f}초. 세 실행의 부모 프로세스 최대 RSS는
  {max(r["peak_process_rss_bytes_macos"] for r in resource) / 1e6:.1f}MB였다. 스레드 1개, GPU 없음.
  wall time이며 전용 장비에서의 엄밀한 CPU 누적 시간은 아니다.
- R1에서 이미 본 test에 대한 **사전 등록된 반복 ablation**이다. 새 독립 test라고 부르지 않는다.
  공식 모델의 알려지지 않은 학습 노출 문제도 해결한 것이 아니다.

재현 순서: `python -m experiments.followup.train_curve`, `evaluate_curve`, `stability`,
`final_audit`, `report_r2`. `.venv-research/bin/python`을 사용하고 기존 artifacts를 보존한다.
모든 모델과 행렬은 `models/learning-curve/`, 원본 checkpoint·점수·변형 특징은
`artifacts/followup-v2/R2/`에 있다. 기존 출력이 있으면 별도 OUT 경로를 쓴 checkout에서 재실행한다.
"""
    (ROOT / "FOLLOWUP_R2_REPORT.md").write_text(report)
    print("R2 report written", decision, flush=True)


if __name__ == "__main__":
    main()
