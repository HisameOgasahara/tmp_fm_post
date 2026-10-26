from dataclasses import asdict, dataclass, field
import math


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "auto"
    cpu_threads: int = 4
    clusters: int = 8
    ring_radius: float = 2.0
    cluster_std: float = 0.12
    target_angle: float = math.pi / 4
    target_radius: float = 0.35
    reward_width: float = 0.6
    outside_sigma: float = 3.0
    hidden_width: int = 128
    hidden_layers: int = 3
    time_frequencies: int = 4
    steps: int = 32
    batch_size: int = 256
    eval_samples: int = 2048
    path_samples: int = 12
    grid_size: int = 25
    plot_limit: float = 3.5
    field_times: tuple[float, ...] = (0.1, 0.5, 0.9)
    arrow_scale: float = 16.0
    delta_arrow_scale: float = 8.0
    direction_arrow_scale: float = 12.0
    plot_dpi: int = 110
    animation_fps: int = 5
    animation_embed_mb: float = 100.0
    pretrain_updates: int = 2500
    pretrain_lr: float = 1e-3
    updates: int = 100
    checkpoint_every: int = 10
    learning_rates: dict = field(default_factory=lambda: {
        "RL": 1e-4, "Direct": 1e-4, "DPO": 1e-4})
    regularization: dict = field(default_factory=lambda: {
        "RL": 0.0, "Direct": 0.0, "DPO": 0.0})
    grad_clip: float = 1.0
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    advantage_normalization: str = "std"
    sde_epsilon: float = 0.02
    sde_noise_schedule: str = "flow_grpo"
    sde_noise: float = 0.7
    dpo_beta: float = 2.0
    dpo_timesteps: int | None = None
    candidate_count: int = 4096
    preference_pairs: int = 2048
    preference_tie_tolerance: float = 0.0
    coverage_exclusion_radius: float = 0.7
    seed_offsets: dict = field(default_factory=lambda: {
        "evaluation": 10000, "training": 20000, "preferences": 30000,
        "one_update": 40000})

    def validate(self):
        positive = ("clusters", "hidden_width", "hidden_layers", "steps",
                    "batch_size", "eval_samples", "path_samples", "grid_size",
                    "checkpoint_every", "candidate_count", "preference_pairs",
                    "ppo_epochs", "cpu_threads", "time_frequencies")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.grid_size < 2 or not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("grid_size >= 2 and unique nonempty seeds required")
        if not 0 < self.sde_epsilon < 0.5 or self.sde_noise <= 0:
            raise ValueError("SDE requires 0 < epsilon < 0.5 and positive noise")
        if self.sde_noise_schedule not in ("flow_grpo", "constant"):
            raise ValueError("sde_noise_schedule must be 'flow_grpo' or 'constant'")
        if self.advantage_normalization not in ("std", "mean"):
            raise ValueError("advantage_normalization must be 'std' or 'mean'")
        if self.dpo_beta <= 0 or (self.dpo_timesteps is not None and self.dpo_timesteps <= 0):
            raise ValueError("dpo_beta and dpo_timesteps must be positive")
        if min(self.target_radius, self.reward_width, self.cluster_std) <= 0:
            raise ValueError("Radii, width and standard deviation must be positive")
        if self.updates < 0 or self.pretrain_updates < 0:
            raise ValueError("Update counts cannot be negative")
        if self.eval_samples < self.path_samples:
            raise ValueError("eval_samples must cover path_samples")
        for method in ("RL", "Direct", "DPO"):
            if self.learning_rates[method] <= 0 or self.regularization[method] < 0:
                raise ValueError("Invalid optimizer configuration")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def smoke(cls):
        """실행 연결 확인용 축소 설정. 실험 결론에는 기본 설정을 사용한다."""
        return cls(seeds=(0,), hidden_width=32, pretrain_updates=15,
                   updates=2, checkpoint_every=1, steps=8, batch_size=32,
                   eval_samples=64, path_samples=6, grid_size=9,
                   candidate_count=128, preference_pairs=64, ppo_epochs=2)
