import math
import torch
from torch import nn


def select_device(config):
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VelocityNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("frequencies", 2 ** torch.arange(config.time_frequencies).float() * math.pi)
        layers = []
        width = 3 + 2 * config.time_frequencies
        for _ in range(config.hidden_layers):
            layers.extend((nn.Linear(width, config.hidden_width), nn.SiLU()))
            width = config.hidden_width
        layers.append(nn.Linear(width, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        t = t.expand(x.shape[0]).reshape(-1, 1) if t.ndim == 0 else t.reshape(-1, 1)
        phase = t * self.frequencies
        return self.network(torch.cat((x, t, phase.sin(), phase.cos()), dim=-1))


def cluster_centers(config, device):
    angles = torch.arange(config.clusters, device=device) * (2 * math.pi / config.clusters)
    return config.ring_radius * torch.stack((angles.cos(), angles.sin()), dim=-1)


def sample_data(config, count, device):
    centers = cluster_centers(config, device)
    indices = torch.randint(config.clusters, (count,), device=device)
    return centers[indices] + config.cluster_std * torch.randn(count, 2, device=device)


def target_center(config, device):
    return torch.tensor([math.cos(config.target_angle), math.sin(config.target_angle)],
                        device=device) * config.ring_radius


def compute_reward(x, config, kind="smooth", gradient_access=True):
    distance2 = (x - target_center(config, x.device)).square().sum(-1)
    if kind == "smooth":
        value = torch.exp(-distance2 / (2 * config.reward_width ** 2))
    elif kind == "binary":
        value = (distance2 <= config.target_radius ** 2).to(x.dtype)
    else:
        raise ValueError(f"Unknown reward: {kind}")
    return value if gradient_access else value.detach()


def integrate_ode(model, noise, steps, retain_grad=False):
    x = noise.clone()
    if retain_grad:
        x.requires_grad_(True)
        x.retain_grad()
    states = [x]
    for step in range(steps):
        x = x + model(x, step / steps) / steps
        if retain_grad:
            x.retain_grad()
        states.append(x)
    return x, states


def estimate_score(model, x, t):
    """x_t=(1-t)z+t*y에서 score=(t*v-x)/(1-t), 0<t<1."""
    return (t * model(x, t) - x) / (1 - t)


def sde_sigma(t, config):
    """SDE 확산 계수 σ(t).

    flow_grpo: Flow-GRPO의 g(s)^2 = a^2 s/(1-s)를 잡음 수준 s = 1 - t로 옮긴
    σ(t) = a·sqrt((1-t)/t). 잡음 쪽(t→0)에서 크고 데이터 쪽(t→1)에서 0으로 줄어든다.
    constant: σ(t) = a.
    """
    if config.sde_noise_schedule == "constant":
        return torch.full_like(t, config.sde_noise) if torch.is_tensor(t) else config.sde_noise
    if torch.is_tensor(t):
        return config.sde_noise * torch.sqrt((1 - t) / t)
    return config.sde_noise * math.sqrt((1 - t) / t)


def transition_mean(model, x, t, dt, sigma, velocity=None):
    """dx = [v + σ²/2·score] dt + σ dW 의 Euler–Maruyama 평균. ODE와 같은 주변분포를 갖는다."""
    v = model(x, t) if velocity is None else velocity
    score = (t * v - x) / (1 - t)
    return x + (v + 0.5 * sigma ** 2 * score) * dt


def gaussian_log_prob(next_x, mean, std):
    return (-(next_x - mean).square() / (2 * std ** 2) - torch.log(std)
            - 0.5 * math.log(2 * math.pi)).sum(-1)


def transition_log_prob(model, x, next_x, t, dt, sigma):
    mean = transition_mean(model, x, t, dt, sigma)
    std = torch.as_tensor(sigma * math.sqrt(dt), device=x.device, dtype=x.dtype)
    return gaussian_log_prob(next_x, mean, std.expand_as(mean))


def flatten_rollout(rollout, config):
    """수집 경로의 모든 내부 전이를 (단계×샘플) 한 배치로 펼친다."""
    states = rollout["states"]
    steps, batch = states.shape[0] - 1, states.shape[1]
    x = states[:-1].reshape(-1, 2)
    next_x = states[1:].reshape(-1, 2)
    t = torch.tensor(rollout["times"], device=x.device, dtype=x.dtype)
    t = t[:, None].expand(steps, batch).reshape(-1, 1)
    return x, next_x, t, sde_sigma(t, config), (steps, batch)


def rollout_log_probs(model, rollout, config, velocity=None):
    """모든 단계의 전이 log-prob을 한 번의 모델 호출로 계산한다. 반환 형태 (단계, 샘플)."""
    x, next_x, t, sigma, shape = flatten_rollout(rollout, config)
    v = model(x, t) if velocity is None else velocity
    mean = transition_mean(model, x, t, rollout["dt"], sigma, velocity=v)
    std = (sigma * math.sqrt(rollout["dt"])).expand_as(mean)
    return gaussian_log_prob(next_x, mean, std).reshape(shape), v


def transition_kl(v, v_ref, x, t, sigma, dt):
    """같은 분산의 두 가우시안 전이 사이 닫힌형 KL = ||μθ - μref||² / (2σ²dt).

    μθ - μref = (vθ - vref)(1 + σ²t / (2(1-t))) dt 이므로 속도 차이만으로 계산된다.
    """
    factor = 1 + sigma ** 2 * t / (2 * (1 - t))
    return (dt / 2) * (factor / sigma).square().squeeze(-1) * (v - v_ref).square().sum(-1)


@torch.no_grad()
def collect_sde(model, noise, config, step_noise=None):
    """ODE로 양 끝을 연결하고 내부 구간에서 Euler–Maruyama 전이를 수집한다."""
    eps = config.sde_epsilon
    dt = (1 - 2 * eps) / config.steps
    x = noise + eps * model(noise, 0.0)
    states, log_probs = [x], []
    times = [eps + i * dt for i in range(config.steps)]
    if step_noise is None:
        step_noise = torch.randn(config.steps, *noise.shape, device=noise.device)
    for i, t in enumerate(times):
        sigma = sde_sigma(t, config)
        next_x = transition_mean(model, x, t, dt, sigma)
        next_x = next_x + sigma * math.sqrt(dt) * step_noise[i]
        log_probs.append(transition_log_prob(model, x, next_x, t, dt, sigma))
        x = next_x
        states.append(x)
    endpoint = x + eps * model(x, 1 - eps)
    return {"states": torch.stack(states), "log_probs": torch.stack(log_probs),
            "times": times, "dt": dt, "endpoint": endpoint,
            "noise": noise, "step_noise": step_noise,
            "path": torch.stack([noise, *states, endpoint])}


def clone_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def build_model(config, state=None):
    model = VelocityNet(config).to(select_device(config))
    if state is not None:
        model.load_state_dict(state)
    return model


def compute_metrics(points, config, reward_kind):
    distances = torch.cdist(points, cluster_centers(config, points.device))
    distance, cluster = distances.min(dim=1)
    outside = distance > config.outside_sigma * config.cluster_std
    cluster = cluster.clone()
    cluster[outside] = config.clusters
    proportions = torch.bincount(cluster, minlength=config.clusters + 1) / len(points)
    metrics = {"mean_reward": compute_reward(points, config, reward_kind).mean().item(),
               "success_rate": compute_reward(points, config, "binary").mean().item(),
               "outside_rate": outside.float().mean().item()}
    metrics.update({f"cluster_{i}": value.item() for i, value in enumerate(proportions)})
    return metrics
