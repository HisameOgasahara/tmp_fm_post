import json
import shutil
import matplotlib.pyplot as plt

from .plots import (export_animations, plot_distributions, plot_fields,
                    plot_learning_curves, plot_one_update, plot_paths, plot_update_details)


def build_report(experiment):
    lines = ["# RF Post-training 실험 결과", "", "## 실행 설정", "",
             f"- 실행 장치: `{experiment.device}`",
             f"- 반복 시드: {list(experiment.config.seeds)}",
             f"- 사전학습: {experiment.config.pretrain_updates}회",
             f"- 추가 학습: {experiment.config.updates}회",
             "- 생성 방향: 0은 잡음 출발, 1은 생성 완료",
             "- 군집 밖 판정: 가장 가까운 군집 중심에서 표준편차의 "
             f"{experiment.config.outside_sigma:g}배보다 먼 점", "",
             "## 학습 신호와 기법 선택", "",
             "| 조건 | 직접 역전파 | 온라인 RL | DPO |",
             "| --- | --- | --- | --- |",
             "| 매끄러운 보상과 기울기 접근 | 사용 가능 | 점수 차이 필요 | 유효 선호 쌍 필요 |",
             "| 매끄러운 점수만 제공 | 기울기 접근 차단으로 건너뜀 | 점수 차이 필요 | 유효 선호 쌍 필요 |",
             "| 성공 여부 보상 | 유용한 보상 기울기 없음 | 성공·실패 차이 필요 | 성공·실패 쌍 필요 |",
             "| 새 학습용 평가 불가, 저장된 선호 존재 | 제외 | 제외 | 저장된 기록으로 학습 |", "",
             "## 실험별 관찰", ""]
    for name, run in experiment.runs.items():
        methods = sorted({row["method"] for row in run["rows"]})
        lines.extend([f"### {name}", "",
                      f"보상: `{run['reward_kind']}`, 기울기 접근: `{run['gradient_access']}`, "
                      f"목표 반경: {run['config'].target_radius:g}.", "",
                      "| 기법 | 평균 보상 전→후 | 도달률 전→후 | 군집 밖 비율 | 학습용 평가 수 | 학습 시간(초) | 최대 GPU MB |",
                      "| --- | --- | --- | --- | --- | --- | --- |"])
        for method in methods:
            rows = [row for row in run["rows"] if row["method"] == method]
            initial = [row for row in rows if row["step"] == 0]
            final = [row for row in rows if row["step"] == run["config"].updates]
            def mean(records, key):
                return sum(record[key] for record in records) / len(records)
            lines.append(f"| {method} | {mean(initial, 'mean_reward'):.3f} → {mean(final, 'mean_reward'):.3f} | "
                         f"{mean(initial, 'success_rate'):.1%} → {mean(final, 'success_rate'):.1%} | "
                         f"{mean(final, 'outside_rate'):.1%} | {mean(final, 'reward_evaluations'):.0f} | "
                         f"{mean(final, 'training_seconds'):.2f} | {max(r['peak_gpu_mb'] for r in final):.1f} |")
        lines.extend(["", f"![벡터장](figures/{name}/vector_fields.png)",
                      f"![벡터장 변화](figures/{name}/vector_field_changes.png)",
                      f"![생성 경로](figures/{name}/trajectories.png)",
                      f"![도착점 분포](figures/{name}/distributions.png)", "",
                      "벡터장 변화 그림은 같은 위치·생성 시점의 속도 차이를 나타낸다. "
                      "생성 경로와 도착점은 같은 초기 잡음으로 비교한다.", ""])
    lines.extend(["## 비용 해석", "", "![성능과 비용](figures/learning_curves.png)", "",
                  "학습 시간에는 결과 수집·학습 업데이트와 DPO 선호 자료 준비 시간을 포함한다. "
                  "공통 사전학습, 평가, 그림 생성, 파일 저장은 분리한다. "
                  "DPO 후보 평가는 최초 한 번만 계상하며, 재사용한 기록에 대한 추가 학습용 평가는 0이다. "
                  "평가 수가 고정된 DPO 곡선은 같은 가로축 위치에서 세로로 변할 수 있다. "
                  "얇은 선은 시드별 결과이며 굵은 선은 같은 업데이트 횟수에서 비용과 성능을 각각 평균한 값이다.", "",
                  "## 해석 범위", "",
                  "DPO는 RF 예측 오차 차이에 적용한 선호 목적함수이다. "
                  "정확한 ODE 우도 기반 DPO와 동일하다고 해석하지 않는다. "
                  "RL은 시간 내부의 가우시안 SDE 전이 확률을 사용하며, 양 끝 ODE 연결은 "
                  "수집 당시 모델로 고정된 것으로 다룬다. 근사적인 끝단 처리이므로 epsilon과 잡음 크기의 영향을 확인한다. "
                  "RL의 학습 경로는 SDE이고 공통 성능 평가는 ODE이므로 두 결과는 다를 수 있다. "
                  "큰 보상만으로 원래 분포 보존을 보장할 수 없으며, 군집 밖 비율과 군집별 비율을 함께 확인한다.", ""])
    return "\n".join(lines)


def export_all(experiment, animate=True, progress=print):
    if not experiment.runs:
        raise RuntimeError("먼저 실험을 실행하세요.")
    experiment.save_metrics()
    first_name = next(iter(experiment.runs))
    for name in experiment.runs:
        directory = experiment.output / "figures" / name
        for filename, function, options in (
            ("vector_fields.png", plot_fields, {}),
            ("vector_field_changes.png", plot_fields, {"delta": True}),
            ("vector_field_directions.png", plot_fields, {"delta": True, "normalize": True}),
            ("trajectories.png", plot_paths, {}),
            ("distributions.png", plot_distributions, {})):
            fig = function(experiment, name, path=directory / filename, **options)
            plt.close(fig)
            if name == first_name:
                shutil.copy2(directory / filename, experiment.output / "figures" / filename)
        if animate:
            outputs = export_animations(experiment, name, experiment.output / "animations" / name)
            if name == first_name:
                for output in outputs:
                    shutil.copy2(output, experiment.output / "animations" / output.name)
        if progress:
            progress(f"{name}: 그림과 애니메이션 저장 완료" if animate else f"{name}: 그림 저장 완료")
    fig = plot_learning_curves(experiment, path=experiment.output / "figures" / "learning_curves.png")
    plt.close(fig)
    for index, (key, result) in enumerate(experiment.one_updates.items()):
        for filename, function in (("one_update.png", plot_one_update), ("one_update_details.png", plot_update_details)):
            output = experiment.output / "figures" / key / filename
            fig = function(experiment, result, path=output) if function is plot_one_update else function(result, path=output)
            plt.close(fig)
            if index == 0:
                shutil.copy2(output, experiment.output / "figures" / filename)
    (experiment.output / "실험결과.md").write_text(build_report(experiment), encoding="utf-8")
    (experiment.output / "results" / "pretraining.json").write_text(
        json.dumps(experiment.pretraining, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(experiment.output), "zip", root_dir=experiment.output)
    if progress:
        progress(f"내보내기 완료: {archive}")
    return archive
