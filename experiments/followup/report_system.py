"""Audit persisted benchmark ranks/paths and render measured scaling results."""

import hashlib
import json
import shutil
import statistics
import sys

import numpy as np

from experiments.common import ROOT, SEARCH_ROOT, bounded_cpu, read, save, sha

from .common import OUT


def main():
    out = OUT / "R3"
    bounded_cpu(out)
    sys.path.insert(0, str(SEARCH_ROOT / "src"))
    import matplotlib
    from mini3di_search.align_reference import Alignment
    from mini3di_search.records import Alphabet
    from mini3di_search.scoring import Scoring, load_matrix
    from mini3di_search.traceback import rescore_alignment

    from .evaluation import records

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = read(out / "system-results.json")
    data = read(out / "data-freeze.json")
    sc = Scoring(
        load_matrix(
            ROOT / data["protocol"]["matrix"],
            kind=Alphabet.THREE_DI,
            source="fixed released bundle",
            synthetic=False,
        ),
        14,
        2,
    )
    arrays = {}
    for row in data["queries"] + data["targets"]:
        with np.load(out / f"encoded/{row['record_id']}.npz") as a:
            arrays[row["record_id"]] = {"states": a["states"], "mask": a["mask"]}
    rec = {r.record_id: r for r in records(data["queries"] + data["targets"], arrays)}
    audit_checks, eligible_sizes, table = 0, set(), []
    for condition in results["conditions"]:
        raw = read(out / f"benchmark/{condition['name']}-raw.json")
        meta = next(c for c in data["conditions"] if c["name"] == condition["name"])
        ordered = sorted(meta["target_ids"])
        numeric = {sid: i for i, sid in enumerate(ordered)}
        for mode, value in raw.items():
            oracle = value["oracle"]
            reconstructed = []
            cells = 0
            for q in data["queries"]:
                sid = q["record_id"]
                candidates = oracle["candidates"][sid]
                ranked = oracle["rankings"][sid]
                assert ranked == sorted(ranked, key=lambda r: (-r[0], r[1]))
                assert len({r[1] for r in ranked}) == len(ranked)
                assert set(tid for _, tid in ranked).issubset(candidates)
                scored = {tid: score for score, tid in ranked}
                reconstructed.append(
                    (sid, [numeric[t] for t in candidates], [scored.get(t, 0) for t in candidates])
                )
                cells += len(q["aa"]) * sum(len(rec[t].aa) for t in candidates)
            digest = hashlib.sha256(
                json.dumps(reconstructed, separators=(",", ":")).encode()
            ).hexdigest()
            assert digest == oracle["candidate_score_sha256"]
            assert cells == oracle["dp_cells"]
            for path in oracle["paths"]:
                result = Alignment(
                    **{k: v for k, v in path.items() if k not in ["query", "target"]}
                )
                assert (
                    rescore_alignment(
                        rec[path["query"]].sequence(Alphabet.THREE_DI),
                        rec[path["target"]].sequence(Alphabet.THREE_DI),
                        result,
                        sc,
                    )
                    == result.raw_score
                )
            for backend, timings in value["repeats"].items():
                for label, repetitions in timings.items():
                    assert len(repetitions) == 5
                    assert all(r["candidate_score_sha256"] == digest for r in repetitions)
                    reported = condition["modes"][mode]["backends"][backend][label]
                    times = [r["seconds"] for r in repetitions]
                    assert reported["median"] == statistics.median(times)
                    assert reported["minimum"] == min(times) and reported["maximum"] == max(times)
                    audit_checks += len(repetitions)
        mode = condition["modes"]["double-ungapped"]["backends"]
        old, new = mode["baseline"]["top1"], mode["numba_traceback"]["top1"]
        speedup = old["median"] / new["median"]
        if speedup >= 1.5:
            eligible_sizes.add(condition["target_count"])
        full = condition["modes"]["exhaustive"]["backends"]["numba_traceback"]["top1"]["median"]
        table.append(
            f"| {condition['length'][0]}–{condition['length'][1]} | {condition['target_count']} | "
            f"{old['median']:.4f} | {new['median']:.4f} | {speedup:.2f}× | "
            f"{full / new['median']:.2f}× | {condition['retain_exact_at_10']:.3f} |"
        )
    audit = {
        "passed": True,
        "timed_runs_checked": audit_checks,
        "candidate_and_score_digests_reconstructed": True,
        "persisted_oracle_paths_rescored": True,
        "script_sha256": sha(__file__),
        "result_sha256": sha(out / "system-results.json"),
    }
    save(out / "independent-audit.json", audit)
    decision = {
        "equivalence_passed": True,
        "sizes_with_1_5x_speedup": sorted(eligible_sizes),
        "backend_adoption_gate_passed": len(eligible_sizes) >= 2,
        "scope": "Experimental kernels only; production search repository unchanged",
        "filter_quality_is_separate": True,
    }
    save(out / "decisions.json", decision)
    dest = ROOT / "reports/followup-v2/R3"
    dest.mkdir(parents=True, exist_ok=True)
    for name in [
        "system-results.json",
        "data-freeze.json",
        "encoding.json",
        "independent-audit.json",
        "decisions.json",
    ]:
        shutil.copyfile(out / name, dest / name)
    for name in ["execution-freeze.json", "preflight.json", "jit.json"]:
        shutil.copyfile(out / "benchmark" / name, dest / name)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for ax, length in zip(axes, [[60, 120], [121, 250], [251, 500]], strict=True):
        subset = [r for r in results["conditions"] if r["length"] == length]
        for backend, label, color in [
            ("baseline", "Python filters + traceback", "#315c91"),
            ("numba_traceback", "Compiled filters + traceback", "#28836c"),
        ]:
            values = [r["modes"]["double-ungapped"]["backends"][backend]["top1"] for r in subset]
            x = [r["target_count"] for r in subset]
            y = [r["median"] for r in values]
            ax.errorbar(
                x,
                y,
                yerr=[
                    [m - v["minimum"] for m, v in zip(y, values, strict=True)],
                    [v["maximum"] - m for m, v in zip(y, values, strict=True)],
                ],
                marker="o",
                color=color,
                label=label,
                capsize=3,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Distinct targets",
            ylabel="Warm top-1 seconds",
            title=f"Target length {length[0]}–{length[1]}",
        )
    axes[0].legend(fontsize=7)
    for ext in ["png", "svg"]:
        fig.savefig(dest / f"system-scaling.{ext}", dpi=160)
    plt.close(fig)
    skips = "\n".join(
        f"- 길이 {r['length'][0]}–{r['length'][1]}: 5,000개 조건 미실행. "
        f"적격 고유 구조 {r['available']:,}개."
        for r in data["skipped"]
    )
    encoding, jit = read(out / "encoding.json"), read(out / "benchmark/jit.json")
    report = f"""# R3: 같은 검색 결과의 CPU 처리 비용

고정한 10개 query와 실제 구조를 이용해 **{len(results["conditions"])}개 크기·길이 조건**을
실행했다. 5개 backend × 전수/필터 × score-only/top-1 × 5회 반복이며 총
**{audit_checks:,}회**의 측정 결과를 검산했다.
같은 검색 모드 내에서 모든 backend의 후보 ID·정렬 점수·top-1 경로가 같았다.

## 고정 조건과 변경한 부분

기존 배포된 seed29 모델·전용 행렬·gap=(14,2)를 그대로 사용했다. R2에서 새로 선택한
모델과 혼동하지 않는다. query는 R1 validation의 고정 10개다. 처리량용 target은 기존
학습·평가 자료와 겹칠 수 있다. 이 실험으로 생물학적 품질 일반화를 주장하지 않는다.
기존 SCOPe40 압축 파일만 읽었다. query PDB, 같은 AA hash, 같은 원본 파일 hash를 제거하고
모든 대상 규모는 고유 구조의 중첩 부분집합으로 구성했다. 중복 복제로 크기를 늘리지 않았다.

길이 구간은 **target 길이**다. 세 구간별 50/200/1,000개와 혼합 60–500의 5,000개를 실행했다.

{skips}

누적 변경 순서: Python 기준 → Numba double-hit 후보 처리 → 사전 변환된 정수 배열로
Python ungapped 계산 → Numba ungapped 배치 → Numba traceback. 모든 단계는 동일한
exact 3-mer, 동일 대각선의 겹치지 않는 두 hit, window64, ungapped≥20을 유지한다.
정렬의 동점·gap 규칙도 원래 reference와 같다. 300개 무작위/빈 입력/동점 정렬과
35개 반복 문자·무효 mask 후보 사례를 원본 엔진과 별도로 대조했다.

## 실측 결과

각 칸은 5회 중앙값이며 min/max와 score-only, 중간 backend의 값은 JSON에 모두 있다.
backend 효과는 **같은 필터·같은 출력** 비교다. 필터 효과는 최종 backend에서 전수검색을
필터검색으로 바꾼 비율이며, 이때 생기는 검색 손실을 오른쪽에 별도로 표시했다.

| 대상 길이 | 수 | 기준 초 | 최적화 초 | backend 배속 | 전수/필터 배속 | retain@10 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

![동일 결과의 처리 시간](reports/followup-v2/R3/system-scaling.png)

서로 다른 규모 {sorted(eligible_sizes)}에서 1.5배 이상이므로 backend 통과 기준은
**{decision["backend_adoption_gate_passed"]}**다. 이 판단은 계산을 바꾼 효과이며 더 민감한
검색 알고리즘을 만들었다는 뜻이 아니다. retain@10이 0.98 미만인 필터를 높은 보존율의
기본 검색으로 채택하지 않는다. production `mini-3di-search`는 수정하지 않고 검증된
실험 커널을 `experiments/followup/system_kernels.py`에 보존했다.

## 시간·메모리의 범위

- 구조 선정 {data["selection_seconds"]:.1f}초, {encoding["structures"]:,}구조 인코딩·파일 저장
  {encoding["seconds"]:.1f}초. 이 비용은 warm 검색 시간에 포함하지 않는다.
- 첫 JIT 호출 {jit["seconds"]:.3f}초, 기존 전용 cache 파일 {jit["preexisting_cache_files"]}개.
- 검색 반복 sweep {results["wall_seconds"]:.1f}초, 부모 Python peak RSS
  {results["peak_process_rss_bytes_macos"] / 1e6:.1f}MB.
  프로세스 누적 peak이며 시스템 전체 메모리가 아니다.
- 인덱스 생성·packed 변환·배열 준비·출력 시간을 조건별로 분리했다. score-only와 top-1을
  각각 실제 실행했으며 traceback 시간을 빼서 score-only를 추정하지 않았다.
- backend 순서를 반복마다 회전했고 timed sweep은 직렬 실행했다. 단일 CPU 계산 스레드,
  GPU·클라우드 없음. 로컬 단일 장비의 wall-clock 측정이며 다른 하드웨어 속도를 보장하지 않는다.

후보/점수의 digest를 저장된 순위에서 재구성하고, 저장된 top-1 경로를 독립 재채점했으며
모든 반복의 중앙값·범위를 raw 기록에서 검산했다.
[검증 결과](reports/followup-v2/R3/independent-audit.json).
이 결과만으로 GPU 투입의 이득을 주장하지 않는다. 실제 병목과 별도 전송/JIT 비용을 포함한
예비 측정이 필요하다.

재현: `.venv-research/bin/python -m experiments.followup.system_prepare`, `system_benchmark`,
`report_system`. 원본 입력 경로와 환경은 REPRODUCE.md를 따른다. 이미 있는 출력은 보존하고
별도 OUT 경로를 가진 checkout에서 다시 실행한다.
원본 반복 로그는 `artifacts/followup-v2/R3/benchmark/`다.
"""
    (ROOT / "FOLLOWUP_R3_REPORT.md").write_text(report)
    print("R3 persisted-output audit and report complete", audit_checks, decision, flush=True)


if __name__ == "__main__":
    main()
