"""Shared utilities for probe-direction SUBSPACE patching, used when a
target quantity cannot be varied independently of another feature via
input-level counterfactual editing alone -- e.g. cycle-navigation's
true_position, which is a deterministic function of symbol_counts (net
displacement mod 5), so no input edit can change one while holding the
other exactly fixed. Isolates the SPECIFIC linear direction a persisted
linear probe reads, patching only that projection and leaving the
orthogonal complement (which carries symbol_counts-relevant information) at
the corrupt run's own value. This is the resolution already specced in
phase3_counterfactual_design.py's cyclenav position_pairs audit note.

PYTHONPATH=src:analysis python -c "import phase3_subspace_patch_utils"
"""

import numpy as np
import torch


def compute_subspace_projection(coef_matrix, scaler_scale, tol_ratio=1e-6):
    """coef_matrix: [n_classes, H] linear probe weights, fit on
    StandardScaler-transformed features (coef_ operates in standardized
    space: logit = coef_ @ ((x-mean)/scale) + intercept). scaler_scale:
    [H] the same scaler's per-feature scale. Converts coef_ to the RAW
    activation-space direction (d(logit)/dx = coef_/scale) before computing
    the projection, so the subspace patch operates directly on raw
    (unstandardized) activations, matching what the patching hooks read.

    Returns P: [H,H] numpy projection matrix onto the row-space of the
    raw-space-converted coefficient matrix (symmetric, idempotent)."""
    coef_raw = np.asarray(coef_matrix) / np.asarray(scaler_scale)[None, :]
    U, S, Vt = np.linalg.svd(coef_raw, full_matrices=False)
    tol = tol_ratio * S[0]
    rank = int((S > tol).sum())
    V = Vt[:rank]
    P = V.T @ V
    return P


def load_probe_subspace(path):
    """Loads a persisted linear-probe checkpoint (as saved by
    persist_probe_weights in the Phase 2 scripts) and returns its raw-
    activation-space subspace projection matrix P, plus metadata."""
    d = torch.load(path, weights_only=False)
    P = compute_subspace_projection(d["coef"], d["scaler_scale"])
    return P, d


def mamba_patch_relative_subspace(h, clean_idx, corrupt_idx, layer, t_clean, t_corrupt, P):
    _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
    cv = clean_cache[layer][t_clean].to(h.device)
    P_t = torch.as_tensor(P, dtype=torch.float32, device=h.device)

    def hook(module, inputs, output):
        out = output.clone()
        corrupt_vec = out[0, t_corrupt, :]
        delta = cv - corrupt_vec
        out[0, t_corrupt, :] = corrupt_vec + delta @ P_t
        return out

    handle = h.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            rec, _, _ = h.iface.get_logits(h.model, h._model_input(corrupt_idx))
    finally:
        handle.remove()
    return float(rec.item())


def transformer_patch_relative_subspace(h, clean_idx, corrupt_idx, layer, t_clean, t_corrupt, P):
    _, clean_cache = h.run_with_cache(clean_idx, layers=[layer])
    cv = clean_cache[layer][t_clean + h.bos_off].to(h.device)
    row = t_corrupt + h.bos_off
    P_t = torch.as_tensor(P, dtype=torch.float32, device=h.device)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        out = output[0] if is_tuple else output
        out = out.clone()
        corrupt_vec = out[0, row, :]
        delta = cv - corrupt_vec
        out[0, row, :] = corrupt_vec + delta @ P_t
        return (out,) + output[1:] if is_tuple else out

    handle = h.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            rec, _, _ = h.iface.get_logits(h.model, h._model_input(corrupt_idx))
    finally:
        handle.remove()
    return float(rec.item())
