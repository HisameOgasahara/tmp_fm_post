import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.patches import Circle
import torch

from .core import (build_model, cluster_centers, compute_reward, integrate_ode,
                   target_center)


METHOD_COLORS = {"Pretrained": "#777777", "RL": "#2979b8", "Direct": "#dc6a24", "DPO": "#36916d"}


def as_array(tensor):
    return tensor.detach().cpu().numpy()


def make_grid(config, device):
    axis = torch.linspace(-config.plot_limit, config.plot_limit, config.grid_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def draw_background(ax, config, reward_kind="smooth", reward_map=True):
    centers = as_array(cluster_centers(config, "cpu"))
    target = as_array(target_center(config, "cpu"))
    if reward_map:
        axis = torch.linspace(-config.plot_limit, config.plot_limit, config.grid_size * 3)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        values = compute_reward(points, config, reward_kind).reshape(len(axis), len(axis))
        ax.imshow(as_array(values), origin="lower", extent=[-config.plot_limit, config.plot_limit] * 2,
                  cmap="YlOrBr", alpha=0.25, vmin=0, vmax=1)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", c="#666666", s=24)
    ax.add_patch(Circle(target, config.target_radius, fill=False, color="#b43c46", linewidth=1.3))
    ax.set(xlim=(-config.plot_limit, config.plot_limit), ylim=(-config.plot_limit, config.plot_limit),
           aspect="equal")
    ax.grid(alpha=0.12)


def save_figure(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=fig.dpi, bbox_inches="tight")
    return fig


def plot_data(config):
    from .core import sample_data
    # 데이터 미리보기는 이후 학습 난수열을 바꾸지 않는다.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seeds[0])
        points = as_array(sample_data(config, config.eval_samples, "cpu"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, dpi=config.plot_dpi)
    for ax, kind in zip(axes, ("smooth", "binary")):
        draw_background(ax, config, kind)
        ax.scatter(points[:, 0], points[:, 1], s=3, alpha=0.3)
        ax.set_title(f"8 clusters / {kind} reward")
    return fig


def get_models(experiment, name, seed, step=None):
    run = experiment.runs[name]
    config = run["config"]
    models = {"Pretrained": build_model(config, experiment.baselines[seed])}
    for (record_seed, method), history in run["records"].items():
        if record_seed == seed:
            selected = max(history) if step is None else step
            models[method] = build_model(config, history[selected])
    return models


@torch.no_grad()
def plot_fields(experiment, name, seed=None, step=None, delta=False, normalize=False,
                times=None, path=None):
    run = experiment.runs[name]
    config = run["config"]
    seed = config.seeds[0] if seed is None else seed
    times = config.field_times if times is None else times
    models = get_models(experiment, name, seed, step)
    grid = make_grid(config, experiment.device)
    fields = {(method, t): model(grid, t) for method, model in models.items() for t in times}
    methods = [method for method in models if not delta or method != "Pretrained"]
    if delta:
        fields = {(method, t): fields[method, t] - fields["Pretrained", t]
                  for method in methods for t in times}
    common_max = max(float(value.norm(dim=-1).max()) for value in fields.values())
    norm = colors.Normalize(0, max(common_max, 1e-8))
    fig, axes = plt.subplots(len(methods), len(times), squeeze=False,
                             figsize=(4 * len(times), 3.4 * len(methods)),
                             constrained_layout=True, dpi=config.plot_dpi)
    xy = as_array(grid)
    for row, method in enumerate(methods):
        _, states = integrate_ode(models[method], experiment.eval_noise[seed], config.steps)
        for column, t in enumerate(times):
            ax = axes[row, column]
            draw_background(ax, config, run["reward_kind"], reward_map=not delta)
            vector = fields[method, t]
            magnitude = vector.norm(dim=-1)
            if delta:
                ax.imshow(as_array(magnitude.reshape(config.grid_size, config.grid_size)),
                          origin="lower", extent=[-config.plot_limit, config.plot_limit] * 2,
                          cmap="Blues", norm=norm, alpha=0.65)
            arrows = vector / magnitude.clamp_min(1e-8)[:, None] if normalize else vector
            scale = config.direction_arrow_scale if normalize else (
                config.delta_arrow_scale if delta else config.arrow_scale)
            uv = as_array(arrows)
            ax.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles="xy", scale_units="xy",
                      scale=scale, color=METHOD_COLORS[method], alpha=0.8)
            # 표시 시각은 적분 격자에서 선형 보간한다.
            location = t * config.steps
            left = min(math.floor(location), config.steps)
            right = min(left + 1, config.steps)
            points = as_array(states[left] * (1 - (location - left)) + states[right] * (location - left))
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.13, color=METHOD_COLORS[method])
            ax.set_title(f"{method} | t={t:.2f}")
    if delta:
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="Blues"), ax=axes.ravel().tolist(),
                     shrink=0.6, label="Velocity change magnitude")
    fig.suptitle(f"{name} | seed={seed} | step={step if step is not None else config.updates}"
                 + (" | unit directions" if normalize else " | raw vectors"))
    return save_figure(fig, path)


@torch.no_grad()
def plot_paths(experiment, name, seed=None, step=None, path=None):
    run = experiment.runs[name]
    config = run["config"]
    seed = config.seeds[0] if seed is None else seed
    models = get_models(experiment, name, seed, step)
    noise = experiment.eval_noise[seed][:config.path_samples]
    baseline, base_states = integrate_ode(models["Pretrained"], noise, config.steps)
    base = as_array(torch.stack(base_states))
    methods = [method for method in models if method != "Pretrained"]
    fig, axes = plt.subplots(1, len(methods), squeeze=False, figsize=(5 * len(methods), 4.5),
                             constrained_layout=True, dpi=config.plot_dpi)
    for ax, method in zip(axes[0], methods):
        draw_background(ax, config, run["reward_kind"])
        _, states = integrate_ode(models[method], noise, config.steps)
        points = as_array(torch.stack(states))
        for i in range(config.path_samples):
            ax.plot(base[:, i, 0], base[:, i, 1], color=METHOD_COLORS["Pretrained"], alpha=0.5)
            ax.plot(points[:, i, 0], points[:, i, 1], color=METHOD_COLORS[method], alpha=0.8)
        ax.scatter(points[0, :, 0], points[0, :, 1], marker="o", facecolors="none", edgecolors="black", label="start")
        ax.scatter(points[config.steps // 2, :, 0], points[config.steps // 2, :, 1], marker="+", color="black", label="middle")
        ax.scatter(points[-1, :, 0], points[-1, :, 1], marker="*", color=METHOD_COLORS[method], label="end")
        ax.set_title(f"{method} / gray: pretrained")
        ax.legend(fontsize=8)
    return save_figure(fig, path)


@torch.no_grad()
def plot_distributions(experiment, name, seed=None, step=None, path=None):
    from .core import compute_metrics
    run = experiment.runs[name]
    config = run["config"]
    seed = config.seeds[0] if seed is None else seed
    models = get_models(experiment, name, seed, step)
    fig, axes = plt.subplots(2, len(models), squeeze=False, figsize=(4 * len(models), 7),
                             constrained_layout=True, dpi=config.plot_dpi)
    for column, (method, model) in enumerate(models.items()):
        points, _ = integrate_ode(model, experiment.eval_noise[seed], config.steps)
        metrics = compute_metrics(points, config, run["reward_kind"])
        xy = as_array(points)
        draw_background(axes[0, column], config, run["reward_kind"])
        axes[0, column].scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.3, color=METHOD_COLORS[method])
        axes[0, column].set_title(f"{method} | success={metrics['success_rate']:.1%}")
        labels = [str(i) for i in range(config.clusters)] + ["out"]
        axes[1, column].bar(labels, [metrics[f"cluster_{i}"] for i in range(config.clusters + 1)],
                            color=METHOD_COLORS[method])
        axes[1, column].set(ylim=(0, 1), ylabel="Proportion", xlabel="Cluster / outside")
    return save_figure(fig, path)


def plot_learning_curves(experiment, names=None, path=None):
    names = list(experiment.runs) if names is None else names
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, dpi=experiment.config.plot_dpi)
    line_styles = ("-", "--", ":", "-.")
    for experiment_index, name in enumerate(names):
        run = experiment.runs[name]
        methods = sorted({row["method"] for row in run["rows"]})
        for method in methods:
            rows = [row for row in run["rows"] if row["method"] == method]
            seeds = sorted({row["seed"] for row in rows})
            for column, xkey in enumerate(("reward_evaluations", "training_seconds")):
                for line, ykey in enumerate(("mean_reward", "success_rate")):
                    ax = axes[line, column]
                    series = []
                    for seed in seeds:
                        selected = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["step"])
                        xs, ys = [r[xkey] for r in selected], [r[ykey] for r in selected]
                        ax.plot(xs, ys, alpha=0.22, linewidth=0.9, color=METHOD_COLORS[method])
                        series.append((xs, ys))
                    # 같은 업데이트 횟수에서 두 축을 평균한다. DPO의 평가 수 고정은 세로 선으로 보인다.
                    mean_x = [sum(values[0][i] for values in series) / len(series) for i in range(len(series[0][0]))]
                    mean_y = [sum(values[1][i] for values in series) / len(series) for i in range(len(series[0][1]))]
                    ax.plot(mean_x, mean_y, linewidth=2.5, marker=".", color=METHOD_COLORS[method],
                            linestyle=line_styles[experiment_index % len(line_styles)], label=f"{name}/{method}")
                    ax.set(xlabel=xkey, ylabel=ykey, ylim=(-0.02, 1.02))
                    ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.legend(fontsize=7)
    return save_figure(fig, path)


@torch.no_grad()
def plot_one_update(experiment, result, path=None):
    config = result["config"]
    seed = result["seed"]
    reference = build_model(config, experiment.baselines[seed])
    noise = experiment.eval_noise[seed][:config.path_samples]
    before, _ = integrate_ode(reference, noise, config.steps)
    grid = make_grid(config, experiment.device)
    t = config.field_times[len(config.field_times) // 2]
    base_vector = reference(grid, t)
    models = {method: build_model(config, info["state_dict"]) for method, info in result["methods"].items()}
    changes = {method: model(grid, t) - base_vector for method, model in models.items()}
    common_max = max(float(value.norm(dim=-1).max()) for value in changes.values())
    norm = colors.Normalize(0, max(common_max, 1e-8))
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True, dpi=config.plot_dpi)
    for row, (method, info) in enumerate(result["methods"].items()):
        detail = info.get("detail", {})
        ax = axes[row, 0]
        if method == "RL":
            rewards, advantages = as_array(detail["rewards"]), as_array(detail["advantages"])
            ax.plot(rewards, label="terminal reward")
            ax.plot(advantages, label="reward - batch mean")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.legend(fontsize=8)
            ax.set(xlabel="Collected sample", title="RL: terminal feedback")
        elif method == "Direct":
            draw_background(ax, config, result["reward_kind"])
            points = as_array(detail["endpoint"][:config.path_samples])
            gradient = as_array(detail["reward_gradient"][:config.path_samples])
            ax.scatter(points[:, 0], points[:, 1], s=12)
            ax.quiver(points[:, 0], points[:, 1], gradient[:, 0], gradient[:, 1],
                      angles="xy", scale_units="xy", scale=1)
            ax.set_title("Direct: reward gradient at endpoint")
            ax.text(0.02, 0.02, f"mean reward={float(detail['rewards'].mean()):.4f}\n"
                    f"reward -> point -> model: {'ON' if detail['can_differentiate'] else 'OFF'}",
                    transform=ax.transAxes, fontsize=8)
        else:
            draw_background(ax, config, result["reward_kind"])
            if detail:
                for key, marker, color in (("winner", "*", "green"), ("loser", "x", "red"),
                                           ("middle_w", ".", "#6aaf8b"), ("middle_l", ".", "#d89999")):
                    points = as_array(detail[key][:config.path_samples])
                    ax.scatter(points[:, 0], points[:, 1], marker=marker, color=color, label=key)
                ax.legend(fontsize=8)
            ax.set_title(f"DPO: {info['valid_pairs']} stored pairs")
        ax = axes[row, 1]
        draw_background(ax, config, reward_map=False)
        xy, change = as_array(grid), as_array(changes[method])
        ax.imshow(as_array(changes[method].norm(dim=-1).reshape(config.grid_size, config.grid_size)),
                  origin="lower", extent=[-config.plot_limit, config.plot_limit] * 2,
                  cmap="Blues", norm=norm, alpha=0.6)
        base = as_array(base_vector)
        after = as_array(base_vector + changes[method])
        ax.quiver(xy[:, 0], xy[:, 1], base[:, 0], base[:, 1], angles="xy", scale_units="xy",
                  scale=config.arrow_scale, color="gray", alpha=0.35)
        ax.quiver(xy[:, 0], xy[:, 1], after[:, 0], after[:, 1], angles="xy", scale_units="xy",
                  scale=config.arrow_scale, color=METHOD_COLORS[method], alpha=0.8)
        ax.set_title(f"{method}: before (gray) / after, t={t:.2f}")
        ax = axes[row, 2]
        draw_background(ax, config, result["reward_kind"])
        endpoint, _ = integrate_ode(models[method], noise, config.steps)
        initial, moved = as_array(before), as_array(endpoint - before)
        ax.scatter(initial[:, 0], initial[:, 1], color="gray", s=20)
        ax.quiver(initial[:, 0], initial[:, 1], moved[:, 0], moved[:, 1], angles="xy", scale_units="xy",
                  scale=1, color=METHOD_COLORS[method])
        xy = as_array(endpoint)
        ax.scatter(xy[:, 0], xy[:, 1], s=12, color=METHOD_COLORS[method])
        ax.set_title(f"Same ODE starts | updated={info['updated']}")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="Blues"), ax=axes[:, 1].tolist(),
                 shrink=0.45, label="Velocity change magnitude")
    return save_figure(fig, path)


def plot_update_details(result, path=None):
    config = result["config"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, dpi=config.plot_dpi)
    detail = result["methods"]["RL"]["detail"]
    draw_background(axes[0], config, result["reward_kind"])
    for key, color, label in (("path", "gray", "collected"), ("updated_path", METHOD_COLORS["RL"], "updated")):
        points = as_array(detail[key][:, :config.path_samples])
        for i in range(config.path_samples):
            axes[0].plot(points[:, i, 0], points[:, i, 1], color=color, alpha=0.5,
                         label=label if i == 0 else None)
    axes[0].legend()
    axes[0].set_title("SDE paths: identical starts + step noise")
    detail = result["methods"]["Direct"]["detail"]
    axes[1].plot(torch.linspace(0, 1, config.steps + 1).numpy(), as_array(detail["state_gradients"]), marker=".")
    axes[1].set(xlabel="Generation time", ylabel="Mean norm of d(loss)/d(state)", title="Direct: backpropagation signal")
    detail = result["methods"]["DPO"].get("detail")
    if detail:
        for key in ("error_w", "error_l", "ref_error_w", "ref_error_l"):
            axes[2].plot(as_array(detail[key]), label=key, alpha=0.7)
        axes[2].legend(fontsize=8)
    axes[2].set(xlabel="Pair in update batch", ylabel="Flow matching MSE", title="DPO: current / reference errors")
    return save_figure(fig, path)


def export_animations(experiment, name, directory, seed=None):
    run = experiment.runs[name]
    config = run["config"]
    seed = config.seeds[0] if seed is None else seed
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    methods = [method for (s, method) in run["records"] if s == seed]
    steps = sorted(run["records"][(seed, methods[0])])
    grid = make_grid(config, experiment.device)
    xy = as_array(grid)
    noise = experiment.eval_noise[seed]
    with torch.no_grad():
        # 프레임 준비는 평가/시각화 비용이며 학습 비용에 포함하지 않는다.
        training_frames = []
        field_t = config.field_times[len(config.field_times) // 2]
        for step in steps:
            frame = {}
            for method in methods:
                model = build_model(config, run["records"][(seed, method)][step])
                points, _ = integrate_ode(model, noise, config.steps)
                frame[method] = (as_array(model(grid, field_t)), as_array(points))
            training_frames.append(frame)
        generation_frames = []
        final_models = get_models(experiment, name, seed)
        paths = {method: integrate_ode(final_models[method], noise[:config.path_samples], config.steps)[1]
                 for method in methods}
        for i in range(config.steps + 1):
            t = i / config.steps
            generation_frames.append({method: (as_array(final_models[method](grid, t)), as_array(paths[method][i]))
                                      for method in methods})
    paths_out = []
    for filename, frames, labels in (
        ("training_comparison.html", training_frames, [f"update={step}, t={field_t:.2f}" for step in steps]),
        ("generation_paths.html", generation_frames, [f"generation t={i / config.steps:.2f}" for i in range(config.steps + 1)])):
        fig, axes = plt.subplots(1, len(methods), squeeze=False, figsize=(4.5 * len(methods), 4.2),
                                 constrained_layout=True, dpi=config.plot_dpi)
        def draw(index):
            for ax, method in zip(axes[0], methods):
                ax.clear()
                draw_background(ax, config, run["reward_kind"])
                vectors, points = frames[index][method]
                ax.quiver(xy[:, 0], xy[:, 1], vectors[:, 0], vectors[:, 1], angles="xy", scale_units="xy",
                          scale=config.arrow_scale, color=METHOD_COLORS[method], alpha=0.65)
                ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.5, color=METHOD_COLORS[method])
                if filename == "generation_paths.html":
                    for j in range(config.path_samples):
                        trail = [frame[method][1][j] for frame in frames[:index + 1]]
                        ax.plot([p[0] for p in trail], [p[1] for p in trail], color=METHOD_COLORS[method], alpha=0.5)
                ax.set_title(f"{method} | {labels[index]}")
        with plt.rc_context({"animation.embed_limit": config.animation_embed_mb}):
            movie = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / config.animation_fps)
            output = directory / filename
            output.write_text(movie.to_jshtml(), encoding="utf-8")
        plt.close(fig)
        paths_out.append(output)
    return paths_out
