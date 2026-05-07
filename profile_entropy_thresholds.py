import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ddtree as ddtree_module
from ddtree import ddtree_generate
from model import DFlashDraftModel, load_and_process_dataset


def summarize(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p33": float(np.percentile(arr, 33)),
        "p50": float(np.percentile(arr, 50)),
        "p67": float(np.percentile(arr, 67)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-name-or-path", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tree-budget", type=int, default=128)
    parser.add_argument("--json-out", type=str, default="sweep/entropy_profile.json")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    device = torch.device("cuda:0")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    entropy_by_depth: dict[int, list[float]] = {}
    all_entropy: list[float] = []

    original_build_tree = ddtree_module.build_ddtree_tree

    def wrapped_build_tree(
        draft_logits: torch.Tensor,
        budget: int,
        adaptive_branching: bool = False,
        entropy_thresholds: list[float] | None = None,
        branch_k_values: list[int] | None = None,
    ):
        logits = draft_logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_values = entropy.detach().cpu().numpy().tolist()
        for depth, value in enumerate(entropy_values):
            entropy_by_depth.setdefault(depth, []).append(float(value))
            all_entropy.append(float(value))
        return original_build_tree(
            draft_logits,
            budget,
            adaptive_branching,
            entropy_thresholds,
            branch_k_values,
        )

    ddtree_module.build_ddtree_tree = wrapped_build_tree

    try:
        for idx in range(len(dataset)):
            instance = dataset[idx]
            messages = []
            for user_content in instance["turns"]:
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

                response = ddtree_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=draft_model.block_size,
                    tree_budget=args.tree_budget,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    adaptive_branching=False,
                )

                generated_ids = response.output_ids[0, response.num_input_tokens :]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})
    finally:
        ddtree_module.build_ddtree_tree = original_build_tree

    if not all_entropy:
        raise RuntimeError("No entropy values were collected.")

    all_arr = np.array(all_entropy, dtype=np.float64)
    depth_summary: dict[str, Any] = {}
    for depth in sorted(entropy_by_depth.keys()):
        depth_arr = np.array(entropy_by_depth[depth], dtype=np.float64)
        depth_summary[str(depth)] = summarize(depth_arr)

    global_summary = summarize(all_arr)
    thresholds_3_bins = [float(np.percentile(all_arr, 33)), float(np.percentile(all_arr, 67))]
    thresholds_4_bins = [
        float(np.percentile(all_arr, 25)),
        float(np.percentile(all_arr, 50)),
        float(np.percentile(all_arr, 75)),
    ]

    payload = {
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "tree_budget": args.tree_budget,
        "num_entropy_points": int(all_arr.shape[0]),
        "global_summary": global_summary,
        "threshold_recommendations": {
            "3_bins": thresholds_3_bins,
            "4_bins": thresholds_4_bins,
        },
        "per_depth_summary": depth_summary,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Entropy profile complete")
    print(f"  points: {payload['num_entropy_points']}")
    print(
        "  global range/mean:",
        f"[{global_summary['min']:.4f}, {global_summary['max']:.4f}]",
        f"mean={global_summary['mean']:.4f}",
        f"std={global_summary['std']:.4f}",
    )
    print(
        "  recommended thresholds (3 bins):",
        ",".join(f"{x:.4f}" for x in thresholds_3_bins),
    )
    print(
        "  recommended thresholds (4 bins):",
        ",".join(f"{x:.4f}" for x in thresholds_4_bins),
    )
    print(f"  json: {out_path}")


if __name__ == "__main__":
    main()
