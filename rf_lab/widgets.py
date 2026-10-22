import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from .plots import (plot_distributions, plot_fields, plot_one_update, plot_paths,
                    plot_update_details)


def show_one_update_controls(experiment):
    reward = widgets.Dropdown(options=[("매끄러운 보상", "smooth"), ("성공 여부 보상", "binary")],
                              description="보상:")
    access = widgets.ToggleButtons(options=[("기울기 제공", True), ("점수만 제공", False)])
    radius = widgets.FloatSlider(value=experiment.config.target_radius, min=0.01,
                                  max=experiment.config.ring_radius, step=0.01,
                                  description="목표 반경:", continuous_update=False)
    button = widgets.Button(description="초기화 후 한 번 업데이트", button_style="primary",
                             layout=widgets.Layout(width="220px"))
    output = widgets.Output()

    def run(_):
        button.disabled = True
        try:
            with output:
                output.clear_output(wait=True)
                result = experiment.compare_one_update(reward.value, access.value, radius.value)
                for method, info in result["methods"].items():
                    print(f"{method}: {info['status']} | 성공 샘플={info['success_samples']} | "
                          f"유효 선호 쌍={info['valid_pairs']} | 학습률={info['learning_rate']}")
                print("보상 → 생성점 → 모델:", "연결" if reward.value == "smooth" and access.value else "연결 없음")
                for fig in (plot_one_update(experiment, result), plot_update_details(result)):
                    display(fig)
                    plt.close(fig)
        finally:
            button.disabled = False

    button.on_click(run)
    display(widgets.VBox([reward, access, radius, button, output]))
    return {"reward": reward, "access": access, "radius": radius, "button": button, "output": output}


def show_explorer(experiment):
    if not experiment.runs:
        raise RuntimeError("실험 A, B, C 중 하나를 먼저 실행하세요.")
    experiment_name = widgets.Dropdown(options=list(experiment.runs), description="실험:")
    seed = widgets.Dropdown(options=list(experiment.config.seeds), description="시드:")
    step = widgets.SelectionSlider(description="업데이트:", continuous_update=False)
    time = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01,
                                description="생성 시점:", continuous_update=False)
    view = widgets.Dropdown(options=[("벡터장", "field"), ("벡터장 변화", "delta"),
                                     ("생성 경로", "paths"), ("도착점 분포", "distribution")], description="화면:")
    normalize = widgets.Checkbox(value=False, description="단위 방향 화살표")
    output = widgets.Output()
    changing = False

    def set_steps():
        run = experiment.runs[experiment_name.value]
        history = next(history for (s, _), history in run["records"].items() if s == seed.value)
        step.options = sorted(history)
        step.value = max(history)

    def render(change=None):
        nonlocal changing
        if changing:
            return
        changing = True
        try:
            if change and change["owner"] in (experiment_name, seed):
                set_steps()
            with output:
                output.clear_output(wait=True)
                common = (experiment, experiment_name.value, seed.value, step.value)
                if view.value in ("field", "delta"):
                    fig = plot_fields(*common, delta=view.value == "delta", normalize=normalize.value,
                                      times=(time.value,))
                elif view.value == "paths":
                    fig = plot_paths(*common)
                else:
                    fig = plot_distributions(*common)
                display(fig)
                plt.close(fig)
        finally:
            changing = False

    set_steps()
    for control in (experiment_name, seed, step, time, view, normalize):
        control.observe(render, names="value")
    display(widgets.VBox([widgets.HBox([experiment_name, seed]), step, time,
                          widgets.HBox([view, normalize]), output]))
    render()
    return {"experiment": experiment_name, "seed": seed, "step": step, "time": time,
            "view": view, "normalize": normalize, "output": output}
