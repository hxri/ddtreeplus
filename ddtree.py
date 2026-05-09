import heapq
import json
import time
from functools import lru_cache
from types import SimpleNamespace

from loguru import logger
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times


DDTREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")
DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")


_CPP_COMPACT_ENABLED = False


@lru_cache(maxsize=1)
def load_cpp_compact_module():
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:
        logger.warning(f"torch.utils.cpp_extension is unavailable; falling back to Python cache compaction. {exc}")
        return None

    cpp_source = r"""
torch::Tensor compact_tail_inplace(torch::Tensor cache_tensor, int64_t past_length, torch::Tensor keep_current_indices) {
    TORCH_CHECK(cache_tensor.dim() >= 2, "cache_tensor must have rank >= 2");
    TORCH_CHECK(keep_current_indices.dim() == 1, "keep_current_indices must be a 1D tensor");
    TORCH_CHECK(keep_current_indices.scalar_type() == torch::kLong, "keep_current_indices must have dtype torch.long");
    TORCH_CHECK(cache_tensor.device() == keep_current_indices.device(), "cache_tensor and keep_current_indices must be on the same device");

    const int64_t seq_dim = cache_tensor.dim() - 2;
    TORCH_CHECK(past_length >= 0, "past_length must be non-negative");
    TORCH_CHECK(past_length <= cache_tensor.size(seq_dim), "past_length exceeds cache sequence length");

    const int64_t current_length = cache_tensor.size(seq_dim) - past_length;
    if (current_length <= 0) {
        return cache_tensor;
    }

    const int64_t keep_count = keep_current_indices.numel();
    TORCH_CHECK(keep_count >= 0, "keep_count must be non-negative");
    TORCH_CHECK(keep_count <= current_length, "keep_count exceeds appended window length");

    if (keep_count == 0 || keep_count == current_length) {
        return cache_tensor;
    }

    auto tail = cache_tensor.narrow(seq_dim, past_length, current_length);
    auto kept_tail = tail.index_select(seq_dim, keep_current_indices);
    cache_tensor.narrow(seq_dim, past_length, keep_count).copy_(kept_tail);
    return cache_tensor;
}
"""
    try:
        module = load_inline(
            name="ddtree_compact_tail_ext_v1",
            cpp_sources=[cpp_source],
            functions=["compact_tail_inplace"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Loaded inline C++ tail cache compaction extension for DDTree.")
        return module
    except Exception as exc:
        logger.warning(
            f"Failed to build inline C++ tail cache compaction extension; falling back to Python implementation. {exc}"
        )
        return None


def maybe_enable_cpp_compact(enabled: bool) -> None:
    global _CPP_COMPACT_ENABLED
    _CPP_COMPACT_ENABLED = enabled
    if enabled:
        load_cpp_compact_module()


def load_tree_rl_policy(policy_path: str | None) -> dict | None:
    if policy_path is None:
        return None
    with open(policy_path, "r", encoding="utf-8") as handle:
        policy = json.load(handle)
    return policy


def _build_rl_width_profiles(
    depth_limit: int,
    budget: int,
    vocab_size: int,
    entropy_cpu: torch.Tensor,
    base_width: int,
    max_width: int,
) -> tuple[list[str], list[list[int]], list[int]]:
    widths: list[list[int]] = []
    names: list[str] = []
    effective_budgets: list[int] = []

    def clamp_width_list(values: list[float]) -> list[int]:
        out = []
        for value in values:
            width = int(round(value))
            width = max(1, min(width, max_width, budget, vocab_size))
            out.append(width)
        return out

    base_profiles: list[tuple[str, list[int]]] = []

    # Flat high-width profile (closest to fixed-width tree).
    base_profiles.append(("flat", clamp_width_list([max_width for _ in range(depth_limit)])))

    # Front-heavy profile to maximize early acceptance probability.
    base_profiles.append(
        (
            "front_heavy",
            clamp_width_list([max(base_width, max_width / (1.0 + 0.6 * depth)) for depth in range(depth_limit)]),
        )
    )

    # More aggressive front-heavy profile.
    base_profiles.append(
        (
            "front_aggressive",
            clamp_width_list([max(base_width, max_width / (1.0 + 1.2 * depth)) for depth in range(depth_limit)]),
        )
    )

    # Entropy-proportional profile.
    entropy_weights = entropy_cpu.clamp_min(0.0)
    if float(entropy_weights.sum().item()) <= 0.0:
        entropy_weights = torch.ones_like(entropy_weights)
    entropy_fractions = (entropy_weights / float(entropy_weights.sum().item())).tolist()
    base_profiles.append(
        (
            "entropy_proportional",
            clamp_width_list([
                base_width + (max_width - base_width) * frac for frac in entropy_fractions
            ]),
        )
    )

    # Sharpened entropy profile to focus on uncertain depths.
    entropy_sharp = entropy_weights.pow(2.0)
    if float(entropy_sharp.sum().item()) <= 0.0:
        entropy_sharp = torch.ones_like(entropy_sharp)
    entropy_sharp_fractions = (entropy_sharp / float(entropy_sharp.sum().item())).tolist()
    base_profiles.append(
        (
            "entropy_sharp",
            clamp_width_list([
                base_width + (max_width - base_width) * frac for frac in entropy_sharp_fractions
            ]),
        )
    )

    # Add effective tree-budget choices per width profile.
    budget_fractions = [0.5, 0.75, 1.0]
    for profile_name, profile_widths in base_profiles:
        for frac in budget_fractions:
            eff_budget = max(1, min(budget, int(round(budget * frac))))
            names.append(f"{profile_name}_b{frac:.2f}")
            widths.append(profile_widths)
            effective_budgets.append(eff_budget)

    return names, widths, effective_budgets


def _compute_rl_features(
    entropy_cpu: torch.Tensor,
    target_hidden: torch.Tensor | None,
    budget: int,
    depth_limit: int,
    prev_acceptance: float,
    prev_latency_ms: float,
) -> np.ndarray:
    entropy_np = entropy_cpu.cpu().numpy()
    entropy_mean = float(np.mean(entropy_np))
    entropy_std = float(np.std(entropy_np))
    entropy_first = float(entropy_np[0]) if entropy_np.shape[0] > 0 else 0.0
    entropy_last = float(entropy_np[-1]) if entropy_np.shape[0] > 0 else 0.0
    entropy_slope = float((entropy_last - entropy_first) / max(depth_limit - 1, 1))

    target_uncertainty = 0.0
    latent_delta = 0.0
    if target_hidden is not None and target_hidden.numel() > 0:
        frontier = target_hidden[:, -1, :].float()
        frontier_mean_abs = float(frontier.abs().mean().item())
        target_uncertainty = float(frontier.std(unbiased=False).item() / (frontier_mean_abs + 1e-6))
        if target_hidden.shape[1] > 1:
            prev_frontier = target_hidden[:, -2, :].float()
            latent_delta = float(
                (frontier - prev_frontier).norm(dim=-1).mean().item()
                / (frontier.norm(dim=-1).mean().item() + 1e-6)
            )

    features = np.array([
        entropy_mean,
        entropy_std,
        entropy_first,
        entropy_last,
        entropy_slope,
        float(target_uncertainty),
        float(latent_delta),
        float(budget),
        float(depth_limit),
        float(prev_acceptance),
        float(prev_latency_ms),
    ], dtype=np.float32)
    return features


def _select_rl_action(
    policy: dict | None,
    feature_vector: np.ndarray,
    action_count: int,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[int, list[float]]:
    scores = np.zeros(action_count, dtype=np.float32)
    features = feature_vector
    if policy is not None:
        policy_type = str(policy.get("policy_type", "linear_contextual_bandit"))
        feature_mean = np.asarray(policy.get("feature_mean", []), dtype=np.float32)
        feature_std = np.asarray(policy.get("feature_std", []), dtype=np.float32)
        if feature_mean.ndim == 1 and feature_std.ndim == 1 and feature_mean.shape[0] == feature_vector.shape[0] and feature_std.shape[0] == feature_vector.shape[0]:
            features = (feature_vector - feature_mean) / (feature_std + 1e-6)
        if policy_type == "mlp_contextual_bandit":
            state_dict = policy.get("state_dict", {})
            w1 = np.asarray(state_dict.get("net.0.weight", []), dtype=np.float32)
            b1 = np.asarray(state_dict.get("net.0.bias", []), dtype=np.float32)
            w2 = np.asarray(state_dict.get("net.2.weight", []), dtype=np.float32)
            b2 = np.asarray(state_dict.get("net.2.bias", []), dtype=np.float32)
            w3 = np.asarray(state_dict.get("net.4.weight", []), dtype=np.float32)
            b3 = np.asarray(state_dict.get("net.4.bias", []), dtype=np.float32)
            if (
                w1.ndim == 2 and b1.ndim == 1 and
                w2.ndim == 2 and b2.ndim == 1 and
                w3.ndim == 2 and b3.ndim == 1 and
                w1.shape[1] == features.shape[0] and
                w3.shape[0] == action_count
            ):
                hidden1 = np.maximum(w1 @ features + b1, 0.0)
                hidden2 = np.maximum(w2 @ hidden1 + b2, 0.0)
                scores = w3 @ hidden2 + b3
        else:
            weights = np.asarray(policy.get("weights", []), dtype=np.float32)
            bias = np.asarray(policy.get("bias", []), dtype=np.float32)
            if weights.ndim == 2 and weights.shape[0] == action_count and weights.shape[1] == features.shape[0]:
                scores = weights @ features
                if bias.ndim == 1 and bias.shape[0] == action_count:
                    scores = scores + bias

    if float(epsilon) > 0.0 and rng.random() < float(epsilon):
        action = int(rng.integers(0, action_count))
    else:
        action = int(np.argmax(scores))
    return action, scores.tolist()


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
    adaptive_branching: bool = False,
    entropy_thresholds: list[float] | None = None,
    branch_k_values: list[int] | None = None,
    target_hidden: torch.Tensor | None = None,
    # coverage-based: branch to min k that covers >= min_coverage mass
    coverage_branching: bool = False,
    min_coverage: float = 0.8,
    # budget-proportional: k_i = floor(budget * H_i / sum(H))
    budget_proportional_branching: bool = False,
    budget_proportional_alpha: float = 1.0,
    budget_proportional_base_width: int = 1,
    budget_proportional_exact_budget: bool = False,
    budget_proportional_max_width: int | None = None,
    # target-latent-guided: use current target latent as a global uncertainty signal
    target_latent_branching: bool = False,
    target_latent_alpha: float = 1.0,
    target_latent_beta: float = 0.5,
    target_latent_depth_decay: float = 1.0,
    # RL branching: contextual policy selects among width profiles.
    rl_branching: bool = False,
    rl_policy: dict | None = None,
    rl_epsilon: float = 0.0,
    rl_rng: np.random.Generator | None = None,
    rl_prev_acceptance: float = 0.0,
    rl_prev_latency_ms: float = 0.0,
    # draft-prob threshold: branch to tokens with p_draft >= prob_threshold
    prob_threshold_branching: bool = False,
    prob_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float], dict]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
            {},
        )

    vocab_size = int(draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])
    entropy_thresholds = entropy_thresholds or []
    branch_k_values = branch_k_values or [min(budget, vocab_size)]

    # Fixed-width behavior: a single top-k is used at every depth.
    tree_meta: dict = {}

    tree_budget_for_build = budget

    if not adaptive_branching and not coverage_branching and not budget_proportional_branching and not target_latent_branching and not rl_branching and not prob_threshold_branching:
        fixed_topk = min(budget, vocab_size)
        branch_widths = [fixed_topk for _ in range(depth_limit)]
    elif adaptive_branching:
        # Adaptive-width behavior: choose k per position from entropy bins.
        logits_for_entropy = draft_logits.float()
        log_probs = torch.log_softmax(logits_for_entropy, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_cpu = entropy.to(device="cpu", dtype=torch.float32).tolist()

        branch_widths = []
        for entropy_value in entropy_cpu:
            bucket_index = 0
            for threshold in entropy_thresholds:
                if entropy_value > threshold:
                    bucket_index += 1
                else:
                    break
            width = branch_k_values[bucket_index]
            width = max(1, min(int(width), budget, vocab_size))
            branch_widths.append(width)

    elif coverage_branching:
        # Nucleus-style: min k whose cumulative probability mass >= min_coverage.
        logits_for_coverage = draft_logits.float()
        probs = torch.softmax(logits_for_coverage, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # [depth, vocab]
        # k = first index where cumsum >= min_coverage, clipped to [1, budget]
        coverage_tensor = (cumsum_probs >= min_coverage).float()
        # argmax gives first True position (0-based), +1 for count
        k_per_pos = coverage_tensor.argmax(dim=-1) + 1  # [depth]
        k_per_pos = k_per_pos.clamp(1, min(budget, vocab_size))
        branch_widths = k_per_pos.to(device="cpu", dtype=torch.long).tolist()
        branch_widths = [int(w) for w in branch_widths]

    elif budget_proportional_branching:
        # Allocate budget proportional to per-position entropy.
        logits_for_entropy = draft_logits.float()
        log_probs = torch.log_softmax(logits_for_entropy, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)  # [depth]
        entropy_cpu = entropy.to(device="cpu", dtype=torch.float32)
        depth_limit_local = int(draft_logits.shape[0])

        # alpha=1.0 reproduces the original proportional weighting.
        alpha = float(max(1e-6, budget_proportional_alpha))
        weights = entropy_cpu.clamp_min(0.0).pow(alpha)
        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        weight_sum = float(weights.sum().item())
        fractions = (weights / weight_sum).tolist()

        max_width = min(budget, vocab_size)
        if budget_proportional_max_width is not None:
            max_width = min(max_width, max(1, int(budget_proportional_max_width)))
        base_width = max(1, int(budget_proportional_base_width))

        if budget_proportional_exact_budget and budget >= depth_limit_local:
            widths = [base_width for _ in range(depth_limit_local)]
            base_total = sum(widths)
            if base_total > budget:
                widths = [1 for _ in range(depth_limit_local)]
                base_total = depth_limit_local

            remaining = max(0, budget - base_total)
            raw_alloc = [fraction * remaining for fraction in fractions]
            add_floor = [int(value) for value in raw_alloc]
            widths = [widths[i] + add_floor[i] for i in range(depth_limit_local)]

            leftover = remaining - sum(add_floor)
            if leftover > 0:
                fractional_order = sorted(
                    range(depth_limit_local),
                    key=lambda i: raw_alloc[i] - float(add_floor[i]),
                    reverse=True,
                )
                for i in fractional_order[:leftover]:
                    widths[i] += 1

            widths = [max(1, min(int(width), max_width)) for width in widths]

            # If clamping to max_width left spare budget, refill by entropy rank.
            spare = budget - sum(widths)
            if spare > 0:
                refill_order = sorted(range(depth_limit_local), key=lambda i: fractions[i], reverse=True)
                while spare > 0:
                    progressed = False
                    for i in refill_order:
                        if widths[i] < max_width:
                            widths[i] += 1
                            spare -= 1
                            progressed = True
                            if spare == 0:
                                break
                    if not progressed:
                        break

            branch_widths = widths
        else:
            base_total = base_width * depth_limit_local
            remaining = max(0, budget - base_total)
            branch_widths = []
            for fraction in fractions:
                width = base_width + int(fraction * remaining)
                width = max(1, min(int(width), max_width))
                branch_widths.append(width)

    elif target_latent_branching:
        # Use the current target latent as a global uncertainty signal, while
        # still allocating widths across future depths from draft entropy.
        logits_for_entropy = draft_logits.float()
        log_probs = torch.log_softmax(logits_for_entropy, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        weights = entropy.to(device="cpu", dtype=torch.float32).clamp_min(0.0)
        depth_limit_local = int(draft_logits.shape[0])

        alpha = float(max(1e-6, target_latent_alpha))
        beta = float(max(0.0, target_latent_beta))
        depth_decay = float(max(0.0, target_latent_depth_decay))
        weights = weights.pow(alpha)

        target_uncertainty = 0.0
        if target_hidden is not None and target_hidden.numel() > 0:
            frontier = target_hidden[:, -1, :].float()
            frontier_mean_abs = float(frontier.abs().mean().item())
            latent_dispersion = float(frontier.std(unbiased=False).item() / (frontier_mean_abs + 1e-6))
            latent_delta = 0.0
            if target_hidden.shape[1] > 1:
                prev_frontier = target_hidden[:, -2, :].float()
                latent_delta = float(
                    (frontier - prev_frontier).norm(dim=-1).mean().item()
                    / (frontier.norm(dim=-1).mean().item() + 1e-6)
                )
            target_uncertainty = max(0.0, min(latent_dispersion + latent_delta, 8.0))

        if target_uncertainty > 0.0:
            depth_indices = torch.arange(depth_limit_local, dtype=torch.float32)
            depth_boost = 1.0 + beta * target_uncertainty / (1.0 + depth_decay * depth_indices)
            weights = weights * depth_boost

        if float(weights.sum().item()) <= 0.0:
            weights = torch.ones_like(weights)
        weight_sum = float(weights.sum().item())
        fractions = (weights / weight_sum).tolist()

        max_width = min(budget, vocab_size)
        if budget_proportional_max_width is not None:
            max_width = min(max_width, max(1, int(budget_proportional_max_width)))
        base_width = max(1, int(budget_proportional_base_width))

        if budget_proportional_exact_budget and budget >= depth_limit_local:
            widths = [base_width for _ in range(depth_limit_local)]
            base_total = sum(widths)
            if base_total > budget:
                widths = [1 for _ in range(depth_limit_local)]
                base_total = depth_limit_local

            remaining = max(0, budget - base_total)
            raw_alloc = [fraction * remaining for fraction in fractions]
            add_floor = [int(value) for value in raw_alloc]
            widths = [widths[i] + add_floor[i] for i in range(depth_limit_local)]

            leftover = remaining - sum(add_floor)
            if leftover > 0:
                fractional_order = sorted(
                    range(depth_limit_local),
                    key=lambda i: raw_alloc[i] - float(add_floor[i]),
                    reverse=True,
                )
                for i in fractional_order[:leftover]:
                    widths[i] += 1

            widths = [max(1, min(int(width), max_width)) for width in widths]

            spare = budget - sum(widths)
            if spare > 0:
                refill_order = sorted(range(depth_limit_local), key=lambda i: fractions[i], reverse=True)
                while spare > 0:
                    progressed = False
                    for i in refill_order:
                        if widths[i] < max_width:
                            widths[i] += 1
                            spare -= 1
                            progressed = True
                            if spare == 0:
                                break
                    if not progressed:
                        break

            branch_widths = widths
        else:
            base_total = base_width * depth_limit_local
            remaining = max(0, budget - base_total)
            branch_widths = []
            for fraction in fractions:
                width = base_width + int(fraction * remaining)
                width = max(1, min(int(width), max_width))
                branch_widths.append(width)

    elif rl_branching:
        logits_for_entropy = draft_logits.float()
        log_probs = torch.log_softmax(logits_for_entropy, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy_cpu = entropy.to(device="cpu", dtype=torch.float32)

        max_width = min(budget, vocab_size)
        if budget_proportional_max_width is not None:
            max_width = min(max_width, max(1, int(budget_proportional_max_width)))
        base_width = max(1, int(budget_proportional_base_width))

        profile_names, profile_widths, profile_budgets = _build_rl_width_profiles(
            depth_limit=depth_limit,
            budget=budget,
            vocab_size=vocab_size,
            entropy_cpu=entropy_cpu,
            base_width=base_width,
            max_width=max_width,
        )
        feature_vector = _compute_rl_features(
            entropy_cpu=entropy_cpu,
            target_hidden=target_hidden,
            budget=budget,
            depth_limit=depth_limit,
            prev_acceptance=rl_prev_acceptance,
            prev_latency_ms=rl_prev_latency_ms,
        )
        if rl_rng is None:
            rl_rng = np.random.default_rng(0)
        action_id, action_scores = _select_rl_action(
            policy=rl_policy,
            feature_vector=feature_vector,
            action_count=len(profile_widths),
            epsilon=rl_epsilon,
            rng=rl_rng,
        )
        branch_widths = profile_widths[action_id]
        tree_budget_for_build = profile_budgets[action_id]
        tree_meta = {
            "rl": {
                "action_id": int(action_id),
                "action_name": profile_names[action_id],
                "action_names": profile_names,
                "action_scores": action_scores,
                "features": feature_vector.tolist(),
                "branch_widths": [int(width) for width in branch_widths],
                "effective_budget": int(tree_budget_for_build),
            }
        }

    elif prob_threshold_branching:
        # Branch into tokens with p_draft >= prob_threshold, at least 1.
        logits_for_thresh = draft_logits.float()
        probs = torch.softmax(logits_for_thresh, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # number of tokens above threshold per position
        above = (sorted_probs >= prob_threshold).sum(dim=-1)  # [depth]
        above = above.clamp(1, min(budget, vocab_size))
        branch_widths = above.to(device="cpu", dtype=torch.long).tolist()
        branch_widths = [int(w) for w in branch_widths]

    max_topk = max(branch_widths)

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=max_topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(tree_budget_for_build, dtype=np.int64)
    node_depths_np = np.empty(tree_budget_for_build, dtype=np.int64)
    parents_np = np.empty(tree_budget_for_build + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < tree_budget_for_build:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        current_depth_width = branch_widths[depth - 1]
        if rank + 1 < current_depth_width:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes, tree_meta


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[0, 0, :previous_tree_length, previous_tree_start : previous_tree_start + previous_tree_length] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_verified_tree(child_maps: list[dict[int, int]], posterior: torch.Tensor) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def _compact_appended_window(cache_tensor: torch.Tensor, past_length: int, keep_current_indices: torch.Tensor) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return

    if _CPP_COMPACT_ENABLED:
        module = load_cpp_compact_module()
        if module is not None:
            module.compact_tail_inplace(cache_tensor, past_length, keep_current_indices)
            return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(past_key_values: DynamicCache, past_length: int, keep_current_indices: list[int]) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(keep_current_indices, dtype=torch.long, device=device)
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")


@torch.inference_mode()
def ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    adaptive_branching: bool = False,
    entropy_thresholds: list[float] | None = None,
    branch_k_values: list[int] | None = None,
    coverage_branching: bool = False,
    min_coverage: float = 0.8,
    budget_proportional_branching: bool = False,
    budget_proportional_alpha: float = 1.0,
    budget_proportional_base_width: int = 1,
    budget_proportional_exact_budget: bool = False,
    budget_proportional_max_width: int | None = None,
    target_latent_branching: bool = False,
    target_latent_alpha: float = 1.0,
    target_latent_beta: float = 0.5,
    target_latent_depth_decay: float = 1.0,
    rl_branching: bool = False,
    rl_policy: dict | None = None,
    rl_epsilon: float = 0.0,
    rl_reward_latency_penalty: float = 0.05,
    prob_threshold_branching: bool = False,
    prob_threshold: float = 0.05,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    rl_round_records = []
    round_trees = [] if save_tree_traces else None
    rl_rng = np.random.default_rng(0)
    rl_prev_acceptance = 0.0
    rl_prev_latency_ms = 0.0
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes, tree_meta = build_ddtree_tree(
            draft_logits[0],
            tree_budget,
            adaptive_branching=adaptive_branching,
            entropy_thresholds=entropy_thresholds,
            branch_k_values=branch_k_values,
            target_hidden=target_hidden,
            coverage_branching=coverage_branching,
            min_coverage=min_coverage,
            budget_proportional_branching=budget_proportional_branching,
            budget_proportional_alpha=budget_proportional_alpha,
            budget_proportional_base_width=budget_proportional_base_width,
            budget_proportional_exact_budget=budget_proportional_exact_budget,
            budget_proportional_max_width=budget_proportional_max_width,
            target_latent_branching=target_latent_branching,
            target_latent_alpha=target_latent_alpha,
            target_latent_beta=target_latent_beta,
            target_latent_depth_decay=target_latent_depth_decay,
            rl_branching=rl_branching,
            rl_policy=rl_policy,
            rl_epsilon=rl_epsilon,
            rl_rng=rl_rng,
            rl_prev_acceptance=rl_prev_acceptance,
            rl_prev_latency_ms=rl_prev_latency_ms,
            prob_threshold_branching=prob_threshold_branching,
            prob_threshold=prob_threshold,
        )
        tree_build_elapsed = cuda_time() - tree_build_start
        stage_times["tree_build"] += tree_build_elapsed
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        tree_compile_elapsed = cuda_time() - tree_compile_start
        stage_times["tree_compile"] += tree_compile_elapsed

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        verify_elapsed = cuda_time() - verify_stage_start
        stage_times["verify"] += verify_elapsed

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        if rl_branching and "rl" in tree_meta:
            latency_ms = 1000.0 * (tree_build_elapsed + tree_compile_elapsed + verify_elapsed)
            reward = float(len(accepted_indices) - rl_reward_latency_penalty * latency_ms)
            rl_round_records.append({
                "action_id": int(tree_meta["rl"]["action_id"]),
                "action_name": tree_meta["rl"]["action_name"],
                "features": tree_meta["rl"]["features"],
                "action_scores": tree_meta["rl"]["action_scores"],
                "branch_widths": tree_meta["rl"]["branch_widths"],
                "effective_budget": int(tree_meta["rl"].get("effective_budget", tree_budget)),
                "accepted_length": int(len(accepted_indices)),
                "latency_ms": float(latency_ms),
                "reward": reward,
            })
            rl_prev_acceptance = float(len(accepted_indices))
            rl_prev_latency_ms = float(latency_ms)
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "parents": [int(parent) for parent in parents],
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        rl_round_records=rl_round_records,
        round_trees=round_trees,
    )
