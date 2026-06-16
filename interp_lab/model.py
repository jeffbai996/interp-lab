"""Model loading — GPT-2 small with transformer-lens hooks."""
from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


def load_model(name: str = "gpt2", device: str | None = None) -> HookedTransformer:
    """Load a HookedTransformer in eval mode with grad globally disabled — we only
    ever intervene on activations, never train, so autograd is pure overhead here.

    transformer-lens folds LayerNorm into the adjacent weights and centers them by
    default, which keeps the residual stream clean to read (the whole reason
    patching is surgical). Defaults to GPT-2 small on CUDA if a card is present.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()
    return model
