r"""Instrumented forward hooks and fidelity metrics for downstream analysis.

Captures intermediate generation-module representations (\hat{x}_l, \hat{x}_a, \hat{x}_v)
during inference for downstream fidelity scoring and Centered Kernel Alignment (CKA),
supporting Member 3 (Ashish Kumar Dash)'s evaluation and analysis pipeline.

Core capabilities:
1. Linear CKA (Centered Kernel Alignment) computation between representation matrices.
2. Cosine similarity fidelity (raw and position-averaged).
3. GenerationHookManager: Context manager registering PyTorch forward hooks on
   the 9 time-mixing layers and modality projection layers.
4. evaluate_generation_fidelity: Evaluates test loader across missing cases 0 to 5,
   comparing generated representations against complete-data reference representations.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lenmod.operator import TIME_MIXING_LAYERS


def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    r"""Compute Linear Centered Kernel Alignment (CKA) between two feature matrices.

    Args:
        X: Tensor of shape (N, D1).
        Y: Tensor of shape (N, D2).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar CKA value in [0.0, 1.0].
    """
    if X.dim() > 2:
        X = X.reshape(X.size(0), -1)
    if Y.dim() > 2:
        Y = Y.reshape(Y.size(0), -1)

    X = X.float()
    Y = Y.float()

    # Center representations along sample dimension N
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute HSIC components: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    yt_x = torch.matmul(Y.t(), X)
    hsic_xy = torch.sum(yt_x ** 2)

    xt_x = torch.matmul(X.t(), X)
    hsic_xx = torch.sum(xt_x ** 2)

    yt_y = torch.matmul(Y.t(), Y)
    hsic_yy = torch.sum(yt_y ** 2)

    denominator = torch.sqrt(hsic_xx * hsic_yy) + eps
    cka = (hsic_xy / denominator).item()
    return float(max(0.0, min(1.0, cka)))


def cosine_similarity_fidelity(
    generated: torch.Tensor,
    reference: torch.Tensor,
    dim: int = -1,
) -> Dict[str, float]:
    r"""Compute generation fidelity via cosine similarity against reference features.

    Args:
        generated: Generated feature tensor (N, ..., D) or (N, D, T).
        reference: Ground-truth reference feature tensor of matching shape.
        dim: Dimension along which cosine similarity is computed.

    Returns:
        Dictionary with:
            - 'mean_cosine_sim': Average cosine similarity over all positions and samples.
            - 'flattened_cosine_sim': Sample-level cosine similarity over flattened representations.
    """
    gen_f = generated.float()
    ref_f = reference.float()

    # 1. Position-wise cosine similarity
    pos_cos = F.cosine_similarity(gen_f, ref_f, dim=dim)
    mean_cos = pos_cos.mean().item()

    # 2. Flattened per-sample representation cosine similarity
    gen_flat = gen_f.reshape(gen_f.size(0), -1)
    ref_flat = ref_f.reshape(ref_f.size(0), -1)
    flat_cos = F.cosine_similarity(gen_flat, ref_flat, dim=-1).mean().item()

    return {
        "mean_cosine_sim": float(mean_cos),
        "flattened_cosine_sim": float(flat_cos),
    }


# Map each missing modality case (0 to 5) to the active time-mixing layer(s) and generated modality
CASE_MODALITY_MAP = {
    0: {"missing": ["l"], "layers": {"l": "l_avp"}},
    1: {"missing": ["a"], "layers": {"a": "a_lvp"}},
    2: {"missing": ["v"], "layers": {"v": "v_alp"}},
    3: {"missing": ["l", "a"], "layers": {"l": "l_vp", "a": "a_vp"}},
    4: {"missing": ["l", "v"], "layers": {"l": "l_ap", "v": "v_ap"}},
    5: {"missing": ["a", "v"], "layers": {"a": "a_lp", "v": "v_lp"}},
}


class GenerationHookManager:
    r"""Context manager to attach forward hooks capturing generation module outputs.

    Hooks the 9 time-mixing layers and temporal projection layers on PromptModel.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hook_handles: List[Any] = []
        self.captured_outputs: Dict[str, List[torch.Tensor]] = {}
        self.captured_inputs: Dict[str, List[torch.Tensor]] = {}

    def _register(self):
        self.clear()

        # Hook 9 time-mixing layers
        for layer_name in TIME_MIXING_LAYERS.keys():
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                handle = layer.register_forward_hook(self._make_hook(layer_name))
                self.hook_handles.append(handle)

        # Hook 3 projection layers (reference true features)
        for proj_name in ["proj_l", "proj_a", "proj_v"]:
            if hasattr(self.model, proj_name):
                layer = getattr(self.model, proj_name)
                handle = layer.register_forward_hook(self._make_hook(proj_name))
                self.hook_handles.append(handle)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            if name not in self.captured_outputs:
                self.captured_outputs[name] = []
                self.captured_inputs[name] = []
            # Detach to avoid retaining autograd computation graph
            if isinstance(out, torch.Tensor):
                self.captured_outputs[name].append(out.detach().cpu())
            if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                self.captured_inputs[name].append(inp[0].detach().cpu())

        return hook

    def clear(self):
        """Clear all captured activations."""
        self.captured_outputs.clear()
        self.captured_inputs.clear()

    def remove(self):
        """Remove all active PyTorch hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def get_layer_output(self, layer_name: str) -> Optional[torch.Tensor]:
        """Return concatenated output tensor for a hooked layer across all forward passes."""
        if layer_name not in self.captured_outputs or not self.captured_outputs[layer_name]:
            return None
        return torch.cat(self.captured_outputs[layer_name], dim=0)

    def __enter__(self):
        self._register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def _unpack_batch(batch: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack multimodal inputs from either MOSIData tuple/list or standard dict."""
    if isinstance(batch, dict):
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
    elif isinstance(batch, (list, tuple)):
        # MOSIData returns (batch_X, batch_Y, missing_mod) where batch_X is (text, audio, vision)
        if len(batch) >= 2 and isinstance(batch[0], (list, tuple)) and len(batch[0]) >= 3:
            text = batch[0][0].to(device)
            audio = batch[0][1].to(device)
            vision = batch[0][2].to(device)
        elif len(batch) >= 3:
            text = batch[0].to(device)
            audio = batch[1].to(device)
            vision = batch[2].to(device)
        else:
            raise ValueError(f"Unrecognized batch format with length {len(batch)}")
    else:
        raise ValueError(f"Unrecognized batch type: {type(batch)}")
    return text, audio, vision


def evaluate_generation_fidelity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    r"""Evaluate generation fidelity and CKA across missing modality cases 0 to 5.

    Compares the generated representations (\hat{x}_m) against the complete-data
    reference representations (proj_m(x_m) + prompt_nm) on the provided dataset split.

    Args:
        model: PromptModel instance (baseline or invariant).
        loader: Evaluation DataLoader (typically test split).
        device: Target execution device.

    Returns:
        Structured dictionary containing:
            - 'cases': Per-case fidelity metrics (cosine similarity, CKA) for each missing modality.
            - 'mean_fidelity': Aggregated mean cosine similarity and CKA across all missing cases.
    """
    model.eval()

    # Step 1: Run complete data pass (case 6) to extract ground-truth reference representations
    ref_features: Dict[str, List[torch.Tensor]] = {"l": [], "a": [], "v": []}

    with torch.no_grad():
        for batch in loader:
            text, audio, vision = _unpack_batch(batch, device)
            batch_sz = text.size(0)

            # In complete data mode (case 6), reference features are proj_m(x_m) + prompt_nm
            # We can obtain proj_m directly
            ref_l = model.proj_l(text.transpose(1, 2)) + model.promptl_nm  # (B, d_p, llen)
            ref_a = model.proj_a(audio.transpose(1, 2)) + model.prompta_nm  # (B, d_p, alen)
            ref_v = model.proj_v(vision.transpose(1, 2)) + model.promptv_nm  # (B, d_p, vlen)

            ref_features["l"].append(ref_l.detach().cpu())
            ref_features["a"].append(ref_a.detach().cpu())
            ref_features["v"].append(ref_v.detach().cpu())

    all_ref_l = torch.cat(ref_features["l"], dim=0)  # (N, prompt_dim, llen)
    all_ref_a = torch.cat(ref_features["a"], dim=0)  # (N, prompt_dim, alen)
    all_ref_v = torch.cat(ref_features["v"], dim=0)  # (N, prompt_dim, vlen)
    all_refs = {"l": all_ref_l, "a": all_ref_a, "v": all_ref_v}

    # Step 2: For each missing case 0 to 5, run inference with forward hooks attached
    results_by_case = {}
    total_cos_sims = []
    total_ckas = []

    for code, spec in CASE_MODALITY_MAP.items():
        case_results = {}
        with GenerationHookManager(model) as hook_mgr:
            with torch.no_grad():
                for batch in loader:
                    text, audio, vision = _unpack_batch(batch, device)
                    batch_sz = text.size(0)
                    missing_mod = [code] * batch_sz

                    model(text, audio, vision, missing_mod)

            # Process captured output for each generated modality in this case
            for mod_name, layer_name in spec["layers"].items():
                gen_out = hook_mgr.get_layer_output(layer_name)
                if gen_out is None:
                    continue

                # Upstream convention:
                # time-mixer output is (N, mod_len, prompt_dim)
                # Transpose to (N, prompt_dim, mod_len) and add prompt_m
                gen_out = gen_out.transpose(1, 2)
                prompt_m = getattr(model, f"prompt{mod_name}_m").detach().cpu()
                gen_rep = gen_out + prompt_m  # (N, prompt_dim, mod_len)

                ref_rep = all_refs[mod_name]  # (N, prompt_dim, mod_len)

                # Compute Cosine Fidelity
                cos_metrics = cosine_similarity_fidelity(gen_rep, ref_rep, dim=1)

                # Compute Linear CKA
                cka_val = linear_cka(gen_rep, ref_rep)

                case_results[mod_name] = {
                    "layer": layer_name,
                    "cosine_similarity": cos_metrics["mean_cosine_sim"],
                    "flattened_cosine_similarity": cos_metrics["flattened_cosine_sim"],
                    "linear_cka": cka_val,
                }
                total_cos_sims.append(cos_metrics["mean_cosine_sim"])
                total_ckas.append(cka_val)

        results_by_case[code] = {
            "missing_modalities": spec["missing"],
            "modalities": case_results,
        }

    mean_cos = sum(total_cos_sims) / len(total_cos_sims) if total_cos_sims else 0.0
    mean_cka = sum(total_ckas) / len(total_ckas) if total_ckas else 0.0

    return {
        "cases": results_by_case,
        "mean_generation_fidelity": {
            "cosine_similarity": float(mean_cos),
            "linear_cka": float(mean_cka),
        },
    }
