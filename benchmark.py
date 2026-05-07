import argparse
import random
from itertools import chain
from pathlib import Path

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from model import DFlashDraftModel, load_and_process_dataset
from dflash import dflash_generate
from ddtree import ddtree_generate, maybe_enable_cpp_compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--tree-budget", type=str, default="16,32,64,128,256,512,1024")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--ddtree-adaptive-branching", action="store_true")
    parser.add_argument("--ddtree-entropy-thresholds", type=str, default="0.5,1.5")
    parser.add_argument("--ddtree-branch-k-values", type=str, default="1,3,8")
    # Coverage-based branching
    parser.add_argument("--ddtree-coverage-branching", action="store_true")
    parser.add_argument("--ddtree-min-coverage", type=float, default=0.8)
    # Budget-proportional branching
    parser.add_argument("--ddtree-budget-proportional-branching", action="store_true")
    parser.add_argument("--ddtree-budget-proportional-alpha", type=float, default=1.0)
    parser.add_argument("--ddtree-budget-proportional-base-width", type=int, default=1)
    parser.add_argument("--ddtree-budget-proportional-exact-budget", action="store_true")
    parser.add_argument("--ddtree-budget-proportional-max-width", type=int, default=None)
    # Draft-probability threshold branching
    parser.add_argument("--ddtree-prob-threshold-branching", action="store_true")
    parser.add_argument("--ddtree-prob-threshold", type=float, default=0.05)
    parser.add_argument("--disable-cpp-compact-cache", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    entropy_thresholds = [
        float(entropy_threshold)
        for entropy_threshold in args.ddtree_entropy_thresholds.split(",")
        if entropy_threshold.strip() != ""
    ]
    branch_k_values = [
        int(branch_k_value)
        for branch_k_value in args.ddtree_branch_k_values.split(",")
        if branch_k_value.strip() != ""
    ]
    if args.ddtree_adaptive_branching:
        if len(branch_k_values) != len(entropy_thresholds) + 1:
            raise ValueError(
                "--ddtree-branch-k-values must contain exactly len(--ddtree-entropy-thresholds)+1 values when adaptive branching is enabled"
            )
        if any(branch_k_value <= 0 for branch_k_value in branch_k_values):
            raise ValueError("All values in --ddtree-branch-k-values must be > 0")
        if any(
            entropy_thresholds[index] > entropy_thresholds[index + 1]
            for index in range(len(entropy_thresholds) - 1)
        ):
            raise ValueError("--ddtree-entropy-thresholds must be sorted in non-decreasing order")

    enabled_branch_modes = [
        bool(args.ddtree_adaptive_branching),
        bool(args.ddtree_coverage_branching),
        bool(args.ddtree_budget_proportional_branching),
        bool(args.ddtree_prob_threshold_branching),
    ]
    if sum(enabled_branch_modes) > 1:
        raise ValueError("Enable at most one DDTree branching mode at a time")

    if args.ddtree_budget_proportional_branching:
        if args.ddtree_budget_proportional_alpha <= 0:
            raise ValueError("--ddtree-budget-proportional-alpha must be > 0")
        if args.ddtree_budget_proportional_base_width <= 0:
            raise ValueError("--ddtree-budget-proportional-base-width must be > 0")
        if args.ddtree_budget_proportional_max_width is not None and args.ddtree_budget_proportional_max_width <= 0:
            raise ValueError("--ddtree-budget-proportional-max-width must be > 0")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    maybe_enable_cpp_compact(not args.disable_cpp_compact_cache)

    def has_flash_attn() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False

    installed_flash_attn = has_flash_attn()
    if not installed_flash_attn:
        raise RuntimeError("flash_attn must be installed because the draft DFlash model always uses FlashAttention")

    target_attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"
    draft_attn_implementation = "flash_attention_2"

    if not args.flash_attn and installed_flash_attn:
        logger.warning("DDTree uses a custom tree attention mask on the target model. For compatibility, forcing the target verifier to torch.sdpa.")

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=target_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=draft_attn_implementation,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    tree_budgets = [int(tree_budget) for tree_budget in args.tree_budget.split(",")]
    methods_to_run = ["dflash"]
    method_key_to_tree_budget = {}
    if not args.flash_attn:
        ddtree_method_keys = [f"ddtree_tb{tree_budget}" for tree_budget in tree_budgets]
        methods_to_run.extend(ddtree_method_keys)
        method_key_to_tree_budget.update({f"ddtree_tb{tree_budget}": tree_budget for tree_budget in tree_budgets})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    warmup_input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Warmup"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    warmup_input_ids = tokenizer.encode(warmup_input_text, return_tensors="pt").to(target.device)
    warmup_max_new_tokens = min(args.max_new_tokens, 16)

    _ = dflash_generate(
        model=draft_model,
        target=target,
        input_ids=warmup_input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=warmup_max_new_tokens,
        block_size=1,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
    )
    for method_key in methods_to_run:
        if method_key == "dflash":
            _ = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
        else:
            _ = ddtree_generate(
                model=draft_model,
                target=target,
                input_ids=warmup_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=warmup_max_new_tokens,
                block_size=block_size,
                tree_budget=method_key_to_tree_budget[method_key],
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                adaptive_branching=args.ddtree_adaptive_branching,
                entropy_thresholds=entropy_thresholds,
                branch_k_values=branch_k_values,
                coverage_branching=args.ddtree_coverage_branching,
                min_coverage=args.ddtree_min_coverage,
                budget_proportional_branching=args.ddtree_budget_proportional_branching,
                budget_proportional_alpha=args.ddtree_budget_proportional_alpha,
                budget_proportional_base_width=args.ddtree_budget_proportional_base_width,
                budget_proportional_exact_budget=args.ddtree_budget_proportional_exact_budget,
                budget_proportional_max_width=args.ddtree_budget_proportional_max_width,
                prob_threshold_branching=args.ddtree_prob_threshold_branching,
                prob_threshold=args.ddtree_prob_threshold,
            )

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
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
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            response["baseline"] = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )
            for method_key in methods_to_run:
                if method_key == "dflash":
                    response[method_key] = dflash_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                    )
                else:
                    response[method_key] = ddtree_generate(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        tree_budget=method_key_to_tree_budget[method_key],
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        adaptive_branching=args.ddtree_adaptive_branching,
                        entropy_thresholds=entropy_thresholds,
                        branch_k_values=branch_k_values,
                        coverage_branching=args.ddtree_coverage_branching,
                        min_coverage=args.ddtree_min_coverage,
                        budget_proportional_branching=args.ddtree_budget_proportional_branching,
                        budget_proportional_alpha=args.ddtree_budget_proportional_alpha,
                        budget_proportional_base_width=args.ddtree_budget_proportional_base_width,
                        budget_proportional_exact_budget=args.ddtree_budget_proportional_exact_budget,
                        budget_proportional_max_width=args.ddtree_budget_proportional_max_width,
                        prob_threshold_branching=args.ddtree_prob_threshold_branching,
                        prob_threshold=args.ddtree_prob_threshold,
                    )

            spec_response = response[methods_to_run[-1]]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    run_data = {
        "responses": responses,
        "block_size": block_size,
        "draft_attn_implementation": draft_attn_implementation,
        "target_attn_implementation": target_attn_implementation,
        "args": vars(args),
    }
    
    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(run_data, save_path)


if __name__ == "__main__":
    main()
