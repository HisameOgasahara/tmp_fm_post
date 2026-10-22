from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import csv
import importlib.metadata
import json
import time

import torch

from .config import Config
from .core import (build_model, clone_state, collect_sde, compute_metrics,
                   integrate_ode, seed_all, select_device)
from .training import (build_preference_sets, pretrain_model, update_direct,
                       update_dpo, update_rl)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(item) for item in value]
    return value


class Experiment:
    def __init__(self, config=None, output_dir=None):
        self.config = (config or Config()).validate()
        self.device = select_device(self.config)
        if self.device.type == "cpu":
            torch.set_num_threads(self.config.cpu_threads)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.output = Path(output_dir or Path("outputs") / stamp)
        self.output.mkdir(parents=True, exist_ok=False)
        for folder in ("figures", "animations", "results", "checkpoints"):
            (self.output / folder).mkdir()
        self.baselines, self.eval_noise, self.runs, self.preferences = {}, {}, {}, {}
        self.one_updates = {}
        self.pretraining = {}
        self.save_config()
        print(f"실행 장치: {self.device} | 결과: {self.output}")

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def save_config(self):
        versions = {name: importlib.metadata.version(name)
                    for name in ("torch", "matplotlib", "ipywidgets")}
        payload = {"config": self.config.to_dict(), "device": str(self.device),
                   "versions": versions, "ode_solver": "Euler, t=0..1",
                   "sde_endpoints": "ODE 0->epsilon, SDE epsilon->1-epsilon, ODE ->1",
                   "rl_endpoint_gradient": "endpoint ODE frozen during each PPO update",
                   "dpo_objective": "unweighted flow-matching error preference surrogate"}
        (self.output / "results" / "config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def pretrain(self, progress=print):
        for seed in self.config.seeds:
            seed_all(seed)
            model = build_model(self.config)
            self.synchronize()
            start = time.perf_counter()
            losses = pretrain_model(model, self.config, progress)
            self.synchronize()
            self.baselines[seed] = clone_state(model)
            self.pretraining[seed] = {"losses": losses, "seconds": time.perf_counter() - start}
            seed_all(seed + self.config.seed_offsets["evaluation"])
            self.eval_noise[seed] = torch.randn(self.config.eval_samples, 2, device=self.device)
            torch.save({"state_dict": self.baselines[seed],
                        "eval_noise": self.eval_noise[seed].cpu(),
                        "pretraining": self.pretraining[seed]},
                       self.output / "checkpoints" / f"pretrained_seed{seed}.pt")
            if progress:
                progress(f"시드 {seed}: 공통 출발점 저장 완료")
        return self

    def require_baseline(self):
        if set(self.baselines) != set(self.config.seeds):
            raise RuntimeError("먼저 사전학습 셀을 실행하세요.")

    def prepare_preferences(self, name, reward_kind="smooth", coverage=False, config=None):
        self.require_baseline()
        config = config or self.config
        if name in self.preferences:
            raise ValueError(f"선호 기록 이름이 이미 있습니다: {name}")
        records = {}
        for seed in config.seeds:
            seed_all(seed + config.seed_offsets["preferences"])
            reference = build_model(config, self.baselines[seed]).requires_grad_(False)
            self.synchronize()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            start = time.perf_counter()
            datasets, source = build_preference_sets(reference, config, reward_kind, coverage)
            self.synchronize()
            seconds = time.perf_counter() - start
            memory = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
            records[seed] = {"datasets": datasets, "source": source, "seconds": seconds,
                             "peak_gpu_bytes": memory}
            torch.save(cpu_tree(records[seed]), self.output / "results" / f"preferences_{name}_seed{seed}.pt")
        self.preferences[name] = {"records": records, "config": deepcopy(config),
                                  "reward_kind": reward_kind}
        return records

    def update(self, method, model, reference, optimizer, config, reward_kind,
               gradient_access=True, dataset=None, capture=False):
        if method == "RL":
            return update_rl(model, reference, optimizer, config, reward_kind, capture)
        if method == "Direct":
            return update_direct(model, reference, optimizer, config, reward_kind, gradient_access, capture)
        if method == "DPO":
            return update_dpo(model, reference, optimizer, config, dataset, capture)
        raise ValueError(method)

    @torch.no_grad()
    def evaluate(self, model, seed, config, reward_kind):
        self.synchronize()
        start = time.perf_counter()
        endpoint, _ = integrate_ode(model, self.eval_noise[seed], config.steps)
        metrics = compute_metrics(endpoint, config, reward_kind)
        self.synchronize()
        metrics["evaluation_seconds"] = time.perf_counter() - start
        metrics["evaluation_reward_results"] = len(endpoint)
        return metrics

    def run(self, name, reward_kind="smooth", gradient_access=True,
            preference_name=None, coverage=None, target_radius=None, progress=print):
        self.require_baseline()
        if name in self.runs:
            raise ValueError(f"실험 이름이 이미 있습니다: {name}. 새 이름을 사용하세요.")
        config = replace(self.config, target_radius=self.config.target_radius
                         if target_radius is None else target_radius).validate()
        if coverage is not None and preference_name is None:
            raise ValueError("실험 C는 먼저 저장한 선호 기록이 필요합니다.")
        if preference_name is None:
            preference_name = f"{name}_preferences"
            self.prepare_preferences(preference_name, reward_kind, config=config)
        preferences = self.preferences[preference_name]
        if preferences["reward_kind"] != reward_kind or preferences["config"].to_dict() != config.to_dict():
            raise ValueError("선호 기록과 실행 설정이 일치하지 않습니다.")
        methods = ("DPO",) if coverage is not None else ("RL", "Direct", "DPO")
        run = {"name": name, "reward_kind": reward_kind, "gradient_access": gradient_access,
               "config": deepcopy(config), "coverage": coverage,
               "preference_name": preference_name, "records": {}, "rows": []}
        self.runs[name] = run
        for seed in config.seeds:
            record = preferences["records"][seed]
            dataset = record["datasets"][coverage or "wide"]
            for method in methods:
                seed_all(seed + config.seed_offsets["training"])
                model = build_model(config, self.baselines[seed])
                reference = build_model(config, self.baselines[seed]).requires_grad_(False)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rates[method])
                history = {0: clone_state(model)}
                run["records"][(seed, method)] = history
                elapsed = record["seconds"] if method == "DPO" else 0.0
                evaluations = record["source"]["reward_evaluations"] if method == "DPO" else 0
                peak_memory = record["peak_gpu_bytes"] if method == "DPO" else 0
                eval_seconds, eval_results = 0.0, 0
                info = {"signal": 0.0, "valid_pairs": 0 if dataset is None else len(dataset["winner"]),
                        "success_samples": 0, "updated": False, "loss": 0.0, "status": "학습 전"}
                for step in range(config.updates + 1):
                    if step:
                        self.synchronize()
                        if self.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(self.device)
                        start = time.perf_counter()
                        info = self.update(method, model, reference, optimizer, config, reward_kind,
                                           gradient_access, dataset)
                        if method == "DPO" and dataset is None:
                            info["success_samples"] = record["source"]["success_samples"]
                        self.synchronize()
                        elapsed += time.perf_counter() - start
                        evaluations += info["reward_evaluations"]
                        if self.device.type == "cuda":
                            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(self.device))
                    if step % config.checkpoint_every == 0 or step == config.updates:
                        history[step] = clone_state(model)
                        metrics = self.evaluate(model, seed, config, reward_kind)
                        eval_seconds += metrics.pop("evaluation_seconds")
                        eval_results += metrics.pop("evaluation_reward_results")
                        row = {"experiment": name, "method": method, "seed": seed, "step": step,
                               **metrics, "training_seconds": elapsed, "reward_evaluations": evaluations,
                               "evaluation_seconds": eval_seconds, "evaluation_reward_results": eval_results,
                               "peak_gpu_mb": peak_memory / 1024 ** 2,
                               **{key: info[key] for key in ("signal", "valid_pairs", "success_samples",
                                                           "updated", "loss", "status")}}
                        run["rows"].append(row)
                        torch.save(history[step], self.output / "checkpoints" /
                                   f"{name}_{method}_seed{seed}_step{step}.pt")
                        if progress:
                            progress(f"{name} | seed={seed} {method} {step}/{config.updates} "
                                     f"보상={metrics['mean_reward']:.3f}, 도달률={metrics['success_rate']:.1%}")
        self.save_metrics()
        return run

    def compare_one_update(self, reward_kind="smooth", gradient_access=True, target_radius=None):
        self.require_baseline()
        config = replace(self.config, target_radius=self.config.target_radius
                         if target_radius is None else target_radius).validate()
        seed = config.seeds[0]
        seed_all(seed + config.seed_offsets["preferences"])
        reference = build_model(config, self.baselines[seed]).requires_grad_(False)
        datasets, source = build_preference_sets(reference, config, reward_kind)
        result = {"config": config, "seed": seed, "reward_kind": reward_kind,
                  "gradient_access": gradient_access, "methods": {}, "preference_source": source}
        for method in ("RL", "Direct", "DPO"):
            seed_all(seed + config.seed_offsets["one_update"])
            model = build_model(config, self.baselines[seed])
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rates[method])
            info = self.update(method, model, reference, optimizer, config, reward_kind,
                               gradient_access, datasets["wide"], capture=True)
            info["state_dict"] = clone_state(model)
            info["learning_rate"] = config.learning_rates[method]
            if method == "RL":
                detail = info["detail"]
                paired = collect_sde(model, detail["noise"], config, detail["step_noise"])
                detail["updated_path"] = paired["path"]
            result["methods"][method] = cpu_tree(info)
        key = f"{reward_kind}_{'gradient' if gradient_access else 'scores'}_r{config.target_radius:g}"
        self.one_updates[key] = result
        torch.save(cpu_tree({**result, "config": config.to_dict()}),
                   self.output / "results" / f"one_update_{key}.pt")
        return result

    def save_metrics(self):
        rows = [row for run in self.runs.values() for row in run["rows"]]
        if rows:
            with (self.output / "results" / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        settings = {name: {"config": run["config"].to_dict(), "reward_kind": run["reward_kind"],
                           "gradient_access": run["gradient_access"], "coverage": run["coverage"],
                           "preference_name": run["preference_name"]} for name, run in self.runs.items()}
        (self.output / "results" / "runs.json").write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
