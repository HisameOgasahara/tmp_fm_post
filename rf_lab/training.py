from copy import deepcopy
import torch
import torch.nn.functional as F

from .core import (collect_sde, compute_reward, integrate_ode, sample_data,
                   target_center, transition_log_prob)


def pretrain_model(model, config, progress=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain_lr)
    device = next(model.parameters()).device
    losses = []
    for step in range(config.pretrain_updates):
        data = sample_data(config, config.batch_size, device)
        noise = torch.randn_like(data)
        t = torch.rand(len(data), 1, device=device)
        middle = (1 - t) * noise + t * data
        loss = F.mse_loss(model(middle, t), data - noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if progress and ((step + 1) % max(1, config.pretrain_updates // 10) == 0):
            progress(f"사전학습 {step + 1}/{config.pretrain_updates}: loss={loss.item():.4f}")
    return losses


def make_preferences(candidates, scores, config):
    """동점 제외 후 우열이 있는 쌍을 균일한 후보 쌍 분포에서 복원 추출한다."""
    if len(candidates) < 2:
        return None
    sorted_scores, order = torch.sort(scores)
    upper = torch.searchsorted(sorted_scores, sorted_scores + config.preference_tie_tolerance,
                               right=True)
    counts = len(scores) - upper
    total = counts.sum()
    if total == 0:
        return None
    # 각 loser가 가질 수 있는 winner 수로 가중하면 모든 유효 쌍에 같은 확률을 준다.
    low = torch.multinomial(counts.float(), config.preference_pairs, replacement=True)
    high = upper[low] + (torch.rand(config.preference_pairs, device=scores.device) * counts[low]).long()
    return {"winner": candidates[order[high]], "loser": candidates[order[low]],
            "winner_score": sorted_scores[high], "loser_score": sorted_scores[low],
            "available_pair_count": int(total.item())}


@torch.no_grad()
def build_preference_sets(reference, config, reward_kind, coverage=False):
    device = next(reference.parameters()).device
    noise = torch.randn(config.candidate_count, 2, device=device)
    candidates, _ = integrate_ode(reference, noise, config.steps)
    scores = compute_reward(candidates, config, reward_kind)
    successes = compute_reward(candidates, config, "binary")
    datasets = {"wide": make_preferences(candidates, scores, config)}
    if datasets["wide"] is not None:
        datasets["wide"]["success_samples"] = int(successes.sum().item())
    if coverage:
        distance = (candidates - target_center(config, device)).norm(dim=-1)
        keep = distance > config.coverage_exclusion_radius
        datasets["missing"] = make_preferences(candidates[keep], scores[keep], config)
        if datasets["missing"] is not None:
            datasets["missing"]["success_samples"] = int(successes[keep].sum().item())
    return datasets, {"candidates": candidates, "scores": scores, "noise": noise,
                      "reward_evaluations": len(candidates), "success_samples": int(successes.sum().item())}


def apply_step(model, optimizer, loss, config):
    if not torch.isfinite(loss):
        raise FloatingPointError("학습 손실이 유한하지 않습니다. 설정과 학습 신호를 확인하세요.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
    optimizer.step()


def reference_penalty(model, reference, config):
    device = next(model.parameters()).device
    x = sample_data(config, config.batch_size, device)
    z = torch.randn_like(x)
    t = torch.rand(len(x), 1, device=device)
    middle = (1 - t) * z + t * x
    with torch.no_grad():
        target = reference(middle, t)
    return F.mse_loss(model(middle, t), target)


def update_rl(model, reference, optimizer, config, reward_kind, capture=False):
    device = next(model.parameters()).device
    old_model = deepcopy(model).requires_grad_(False)
    noise = torch.randn(config.batch_size, 2, device=device)
    rollout = collect_sde(old_model, noise, config)
    rewards = compute_reward(rollout["endpoint"], config, reward_kind)
    advantages = rewards - rewards.mean()
    info = {"reward_evaluations": len(rewards), "success_samples": int(
        compute_reward(rollout["endpoint"], config, "binary").sum().item()),
        "signal": float(advantages.abs().mean()), "valid_pairs": 0,
        "updated": False, "loss": 0.0,
        "status": "점수 차이 없음" if not torch.any(advantages) else "점수 차이 사용 가능"}
    if torch.any(advantages):
        for _ in range(config.ppo_epochs):
            log_probs = torch.stack([
                transition_log_prob(model, rollout["states"][i], rollout["states"][i + 1],
                                    t, rollout["dt"], config.sde_noise)
                for i, t in enumerate(rollout["times"])])
            ratios = (log_probs - rollout["log_probs"]).exp()
            clipped = ratios.clamp(1 - config.ppo_clip, 1 + config.ppo_clip)
            loss = -torch.minimum(ratios * advantages, clipped * advantages).mean()
            if config.regularization["RL"]:
                loss = loss + config.regularization["RL"] * reference_penalty(model, reference, config)
            apply_step(model, optimizer, loss, config)
            info.update(updated=True, loss=loss.item())
    if capture:
        info["detail"] = {**rollout, "rewards": rewards, "advantages": advantages}
    return info


def update_direct(model, reference, optimizer, config, reward_kind,
                  gradient_access=True, capture=False):
    device = next(model.parameters()).device
    noise = torch.randn(config.batch_size, 2, device=device)
    can_differentiate = gradient_access and reward_kind == "smooth"
    with torch.set_grad_enabled(can_differentiate):
        endpoint, states = integrate_ode(model, noise, config.steps,
                                         retain_grad=capture and can_differentiate)
        rewards = compute_reward(endpoint, config, reward_kind, gradient_access)
        info = {"reward_evaluations": len(rewards), "success_samples": int(
            compute_reward(endpoint.detach(), config, "binary").sum().item()),
            "signal": 0.0, "valid_pairs": 0, "updated": False, "loss": 0.0,
            "status": "기울기 사용 가능" if can_differentiate else (
                "유용한 보상 기울기 없음" if reward_kind == "binary" else "보상 기울기 접근 차단")}
        reward_gradient = torch.zeros_like(endpoint)
        state_gradients = torch.zeros(config.steps + 1, device=device)
        if can_differentiate:
            if capture:
                reward_gradient = torch.autograd.grad(rewards.sum(), endpoint, retain_graph=True)[0]
                # autograd.grad가 retain_grad 버퍼에 남긴 값을 비운 뒤 손실 역전파를 측정한다.
                for state in states:
                    state.grad = None
            loss = -rewards.mean()
            if config.regularization["Direct"]:
                loss = loss + config.regularization["Direct"] * reference_penalty(model, reference, config)
            apply_step(model, optimizer, loss, config)
            if capture:
                state_gradients = torch.stack([s.grad.norm(dim=-1).mean() for s in states])
            gradient_norm = torch.stack([p.grad.square().sum() for p in model.parameters()
                                         if p.grad is not None]).sum().sqrt()
            info.update(updated=True, loss=loss.item(), signal=float(gradient_norm))
    if capture:
        info["detail"] = {"noise": noise, "endpoint": endpoint.detach(),
                          "times": torch.linspace(0, 1, config.steps + 1, device=device),
                          "path": torch.stack([s.detach() for s in states]),
                          "rewards": rewards.detach(), "reward_gradient": reward_gradient.detach(),
                          "state_gradients": state_gradients.detach(),
                          "gradient_access": gradient_access,
                          "can_differentiate": can_differentiate}
    return info


def compute_pair_errors(model, winner, loser, t, noise):
    middle_w = (1 - t) * noise + t * winner
    middle_l = (1 - t) * noise + t * loser
    error_w = (model(middle_w, t) - (winner - noise)).square().mean(-1)
    error_l = (model(middle_l, t) - (loser - noise)).square().mean(-1)
    return error_w, error_l, middle_w, middle_l


def update_dpo(model, reference, optimizer, config, dataset, capture=False):
    info = {"reward_evaluations": 0, "success_samples": 0, "signal": 0.0,
            "valid_pairs": 0 if dataset is None else len(dataset["winner"]),
            "updated": False, "loss": 0.0, "status": "유효한 선호 쌍 없음"}
    if dataset is None:
        return info
    device = next(model.parameters()).device
    index = torch.randint(len(dataset["winner"]), (config.batch_size,), device=device)
    winner, loser = dataset["winner"][index], dataset["loser"][index]
    t = torch.rand(len(index), 1, device=device)
    noise = torch.randn_like(winner)
    ew, el, mw, ml = compute_pair_errors(model, winner, loser, t, noise)
    with torch.no_grad():
        rw, rl, _, _ = compute_pair_errors(reference, winner, loser, t, noise)
    logits = -config.dpo_beta * ((ew - el) - (rw - rl))
    loss = -F.logsigmoid(logits).mean()
    if config.regularization["DPO"]:
        loss = loss + config.regularization["DPO"] * reference_penalty(model, reference, config)
    apply_step(model, optimizer, loss, config)
    info.update(updated=True, loss=loss.item(), success_samples=dataset.get("success_samples", 0),
                signal=float((dataset["winner_score"] -
                dataset["loser_score"]).mean()), status="저장된 선호 쌍 사용 가능")
    if capture:
        info["detail"] = {"winner": winner, "loser": loser, "times": t,
                          "noise": noise, "middle_w": mw, "middle_l": ml,
                          "error_w": ew.detach(), "error_l": el.detach(),
                          "ref_error_w": rw, "ref_error_l": rl, "logits": logits.detach()}
    return info
