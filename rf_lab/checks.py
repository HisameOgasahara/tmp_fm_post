"""Colab에서만 사용자가 실행하는 수치·학습 신호 확인."""

from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path

import torch

from .config import Config
from .core import (build_model, clone_state, collect_sde, compute_reward, estimate_score,
                   integrate_ode, seed_all, target_center, transition_log_prob)
from .training import make_preferences, update_direct, update_dpo, update_rl


def require_colab():
    try:
        import google.colab  # noqa: F401
    except ImportError as error:
        raise RuntimeError("이 확인 코드는 Google Colab에서 실행하세요.") from error


def assert_same_state(before, model):
    assert all(torch.equal(before[key], value.cpu()) for key, value in model.state_dict().items())


def run_signal_checks():
    require_colab()
    config = replace(Config.smoke(), device="cpu")
    seed_all(config.seeds[0])
    model = build_model(config)
    reference = deepcopy(model).requires_grad_(False)
    original = clone_state(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rates["Direct"])
    points = torch.randn(config.batch_size, 2, requires_grad=True)
    on = compute_reward(points, config, "smooth", True)
    off = compute_reward(points, config, "smooth", False)
    assert torch.equal(on.detach(), off) and on.requires_grad and not off.requires_grad
    for kind, access in (("smooth", False), ("binary", True)):
        result = update_direct(model, reference, optimizer, config, kind, access, capture=True)
        assert not result["updated"]
        assert_same_state(original, model)

    # 선호 쌍은 동점을 제외하고 높은 보상을 winner에 배치한다.
    candidates = torch.randn(config.batch_size, 2)
    assert make_preferences(candidates, torch.ones(config.batch_size), config) is None
    scores = torch.arange(config.batch_size).float()
    pairs = make_preferences(candidates, scores, config)
    assert torch.all(pairs["winner_score"] > pairs["loser_score"])
    result = update_dpo(model, reference, optimizer, config, None)
    assert not result["updated"]
    assert_same_state(original, model)

    # 매우 넓은 성공 영역에서 모든 보상이 1이면 RL의 보상 차이 업데이트는 0이다.
    no_difference = replace(config, target_radius=1e6)
    result = update_rl(model, reference, optimizer, no_difference, "binary")
    assert result["signal"] == 0 and not result["updated"]
    assert_same_state(original, model)

    # 같은 정책으로 재계산한 가우시안 전이 확률비는 1이다.
    noise = torch.randn(config.batch_size, 2)
    rollout = collect_sde(model, noise, config)
    log_probs = torch.stack([transition_log_prob(model, rollout["states"][i],
        rollout["states"][i + 1], t, rollout["dt"], config.sde_noise)
        for i, t in enumerate(rollout["times"])])
    assert torch.allclose((log_probs - rollout["log_probs"]).exp(), torch.ones_like(log_probs), atol=1e-6)
    repeated = collect_sde(model, noise, config, rollout["step_noise"])
    assert torch.equal(rollout["path"], repeated["path"])

    # 데이터도 표준 정규분포이면 RF 경로의 분산은 (1-t)^2+t^2이다.
    class GaussianVelocity(torch.nn.Module):
        def forward(self, x, t):
            return (2 * t - 1) * x / ((1 - t) ** 2 + t ** 2)
    t = 0.4
    score = estimate_score(GaussianVelocity(), noise, t)
    assert torch.allclose(score, -noise / ((1 - t) ** 2 + t ** 2), atol=1e-6)

    result = update_direct(model, reference, optimizer, config, "smooth", True, capture=True)
    assert result["updated"] and torch.isfinite(result["detail"]["state_gradients"]).all()
    assert torch.any(result["detail"]["state_gradients"] > 0)
    assert any(not torch.equal(original[key], value.cpu()) for key, value in model.state_dict().items())
    assert_same_state(original, reference)

    # DPO 한 번의 업데이트는 기준 모델을 고정하고 현재 모델을 변경한다.
    model.load_state_dict(original)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rates["DPO"])
    result = update_dpo(model, reference, optimizer, config, pairs, capture=True)
    assert result["updated"] and torch.isfinite(result["detail"]["logits"]).all()
    assert_same_state(original, reference)
    assert any(not torch.equal(original[key], value.cpu()) for key, value in model.state_dict().items())
    print("학습 신호 확인 통과: 기울기 접근, 무신호 건너뛰기, 선호 순서, SDE 확률비·난수 고정, score, 기준 모델 고정")


def run_integration_check():
    require_colab()
    from .experiment import Experiment
    from .export import export_all
    config = Config.smoke()
    with tempfile.TemporaryDirectory(prefix="rf_colab_check_") as temp:
        experiment = Experiment(config, Path(temp) / "run").pretrain(progress=None)
        experiment.compare_one_update()
        experiment.run("A", progress=None)
        experiment.run("A_scores", gradient_access=False, progress=None)
        experiment.run("B", "binary", progress=None)
        experiment.prepare_preferences("C_saved", coverage=True)
        experiment.run("C_wide", preference_name="C_saved", coverage="wide", progress=None)
        experiment.run("C_missing", preference_name="C_saved", coverage="missing", progress=None)
        archive = export_all(experiment, animate=True, progress=None)
        assert Path(archive).is_file()
        required = ("figures/vector_fields.png", "figures/vector_field_changes.png",
                    "figures/one_update.png", "figures/trajectories.png", "figures/distributions.png",
                    "figures/learning_curves.png", "animations/training_comparison.html",
                    "animations/generation_paths.html", "results/metrics.csv", "results/config.json", "실험결과.md")
        for filename in required:
            assert (experiment.output / filename).stat().st_size > 0, filename
        for name in ("A_scores", "B"):
            history = experiment.runs[name]["records"][(config.seeds[0], "Direct")]
            assert all(torch.equal(history[0][key], history[config.updates][key]) for key in history[0])
        for name in ("C_wide", "C_missing"):
            rows = experiment.runs[name]["rows"]
            assert {row["method"] for row in rows} == {"DPO"}
            assert {row["reward_evaluations"] for row in rows} == {config.candidate_count}
    print("축소 통합 확인 통과: A/B/C, 체크포인트, 그림, 애니메이션, CSV, ZIP")
