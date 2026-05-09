#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_records_from_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_records_from_run(path: Path) -> list[dict]:
    run_data = torch.load(path, map_location="cpu", weights_only=False)
    responses = run_data.get("responses", [])
    records = []
    for sample_index, response in enumerate(responses):
        for method_key, method_response in response.items():
            if not method_key.startswith("ddtree_tb"):
                continue
            if not hasattr(method_response, "rl_round_records"):
                continue
            for round_index, round_record in enumerate(method_response.rl_round_records):
                records.append(
                    {
                        "sample_index": int(sample_index),
                        "method_key": method_key,
                        "round_index": int(round_index),
                        **round_record,
                    }
                )
    return records


def gather_records(jsonl_paths: list[str], run_paths: list[str]) -> list[dict]:
    records = []
    for jsonl_path in jsonl_paths:
        for path in sorted(glob.glob(jsonl_path)):
            records.extend(load_records_from_jsonl(Path(path)))
    for run_path in run_paths:
        for path in sorted(glob.glob(run_path)):
            records.extend(load_records_from_run(Path(path)))
    return records


def fit_linear_policy(records: list[dict], ridge_lambda: float) -> dict:
    if not records:
        raise ValueError("No RL records found. Collect data first with --ddtree-rl-branching and exploration.")

    max_action_id = max(int(record["action_id"]) for record in records)
    action_count = max_action_id + 1
    feature_dim = len(records[0]["features"])
    all_features = np.asarray([record["features"] for record in records], dtype=np.float32)
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    weights = np.zeros((action_count, feature_dim), dtype=np.float32)
    bias = np.zeros(action_count, dtype=np.float32)
    action_names = [f"action_{action_id}" for action_id in range(action_count)]

    for action_id in range(action_count):
        action_records = [record for record in records if int(record["action_id"]) == action_id]
        if not action_records:
            continue

        action_names[action_id] = str(action_records[0].get("action_name", action_names[action_id]))

        x = np.asarray([record["features"] for record in action_records], dtype=np.float32)
        x = (x - feature_mean) / feature_std
        y = np.asarray([float(record["reward"]) for record in action_records], dtype=np.float32)

        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        eye = np.eye(x_aug.shape[1], dtype=np.float32)
        coeff = np.linalg.solve(x_aug.T @ x_aug + ridge_lambda * eye, x_aug.T @ y)
        weights[action_id] = coeff[:-1]
        bias[action_id] = coeff[-1]

    return {
        "policy_type": "linear_contextual_bandit",
        "feature_dim": int(feature_dim),
        "action_names": action_names,
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "metadata": {
            "num_records": int(len(records)),
            "ridge_lambda": float(ridge_lambda),
        },
    }


class MLPPolicy(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp_policy(
    records: list[dict],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict:
    if not records:
        raise ValueError("No RL records found. Collect data first with --ddtree-rl-branching and exploration.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    max_action_id = max(int(record["action_id"]) for record in records)
    action_count = max_action_id + 1
    feature_dim = len(records[0]["features"])
    action_names = [f"action_{action_id}" for action_id in range(action_count)]
    for record in records:
        action_names[int(record["action_id"])] = str(record.get("action_name", action_names[int(record["action_id"])]))

    all_features = np.asarray([record["features"] for record in records], dtype=np.float32)
    feature_mean = np.mean(all_features, axis=0)
    feature_std = np.std(all_features, axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    x = (all_features - feature_mean) / feature_std
    action_ids = np.asarray([int(record["action_id"]) for record in records], dtype=np.int64)
    rewards = np.asarray([float(record["reward"]) for record in records], dtype=np.float32)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    model = MLPPolicy(feature_dim=feature_dim, hidden_dim=hidden_dim, action_count=action_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the chosen action's score to match reward via regression.
    for _ in range(epochs):
        optimizer.zero_grad()
        scores = model(x_tensor)
        chosen_scores = scores[torch.arange(scores.shape[0]), torch.tensor(action_ids)]
        loss = torch.mean((chosen_scores - reward_tensor) ** 2)
        loss.backward()
        optimizer.step()

    state_dict = {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}
    return {
        "policy_type": "mlp_contextual_bandit",
        "feature_dim": int(feature_dim),
        "action_names": action_names,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "state_dict": state_dict,
        "hidden_dim": int(hidden_dim),
        "activation": "relu",
        "metadata": {
            "num_records": int(len(records)),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "seed": int(seed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a contextual-bandit policy for DDTree RL branching")
    parser.add_argument("--input-jsonl", action="append", default=[], help="Glob pattern(s) for RL JSONL logs")
    parser.add_argument("--input-runs", action="append", default=[], help="Glob pattern(s) for .pt run files containing rl_round_records")
    parser.add_argument("--output-policy", type=Path, required=True, help="Output JSON policy path")
    parser.add_argument("--ridge-lambda", type=float, default=1.0, help="Ridge regularization coefficient")
    parser.add_argument("--policy-model", choices=["linear", "mlp"], default="linear", help="Policy model family")
    parser.add_argument("--mlp-hidden-dim", type=int, default=32, help="Hidden size for MLP policy")
    parser.add_argument("--mlp-epochs", type=int, default=400, help="Training epochs for MLP policy")
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3, help="Learning rate for MLP policy")
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4, help="Weight decay for MLP policy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    if not args.input_jsonl and not args.input_runs:
        raise ValueError("Provide at least one --input-jsonl or --input-runs source")
    if args.ridge_lambda <= 0:
        raise ValueError("--ridge-lambda must be > 0")
    if args.mlp_hidden_dim <= 0:
        raise ValueError("--mlp-hidden-dim must be > 0")
    if args.mlp_epochs <= 0:
        raise ValueError("--mlp-epochs must be > 0")
    if args.mlp_learning_rate <= 0:
        raise ValueError("--mlp-learning-rate must be > 0")
    if args.mlp_weight_decay < 0:
        raise ValueError("--mlp-weight-decay must be >= 0")

    records = gather_records(args.input_jsonl, args.input_runs)
    if args.policy_model == "linear":
        policy = fit_linear_policy(records, args.ridge_lambda)
    else:
        policy = fit_mlp_policy(
            records,
            hidden_dim=args.mlp_hidden_dim,
            epochs=args.mlp_epochs,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            seed=args.seed,
        )

    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    with args.output_policy.open("w", encoding="utf-8") as handle:
        json.dump(policy, handle, indent=2)

    print(f"Saved policy to {args.output_policy}")
    print(f"Records used: {policy['metadata']['num_records']}")
    print(f"Actions: {len(policy['action_names'])}")


if __name__ == "__main__":
    main()
