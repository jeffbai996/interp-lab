"""The IOI task: the clean/corrupted pair, the named token positions, and the
logit-diff metric.

Notebooks 00–02 each re-established this same setup by hand ("the same
clean/corrupted pair as 01", "the heatmap helper, copied from 01"). Once it had
been copy-pasted a third time it earned a home — the README's own rule. Notebooks
00–02 predate this module and keep their inline copies; 03+ import from here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

# The canonical Wang et al. (2022) example. Corruption swaps the *second* "John"
# (S2) for "Mary" so the answer flips — keeping token alignment identical, which
# is what makes position-wise patching between the two runs well-defined.
CLEAN_PROMPT = "When John and Mary went to the store, John gave a drink to"
CORRUPTED_PROMPT = "When John and Mary went to the store, Mary gave a drink to"


@dataclass
class IOITask:
    """The clean/corrupted pair plus the positions and answer tokens you reach for
    constantly. Positions are fixed for the canonical prompt above (15 tokens incl.
    the BOS that to_tokens prepends)."""

    clean_tokens: torch.Tensor       # [1, pos]
    corrupted_tokens: torch.Tensor   # [1, pos]
    str_tokens: list[str]
    io_tok: int                      # " Mary" — correct answer on the clean prompt
    s_tok: int                       # " John" — the distractor (the subject)
    IO_POS: int = 4                  # " Mary"  (indirect object — the name to copy)
    S1_POS: int = 2                  # first " John"
    S2_POS: int = 10                 # second " John" (duplicated subject; the corrupted slot)
    END: int = 14                    # final " to" (where the next token is predicted)


def ioi_task(model: HookedTransformer) -> IOITask:
    """Build the IOITask for `model` (tokenizes the prompts, resolves answer tokens)."""
    io_tok = model.to_single_token(" Mary")
    s_tok = model.to_single_token(" John")
    clean = model.to_tokens(CLEAN_PROMPT)
    corrupted = model.to_tokens(CORRUPTED_PROMPT)
    assert clean.shape == corrupted.shape, "corruption broke token alignment"
    str_toks = model.to_str_tokens(CLEAN_PROMPT)
    return IOITask(clean, corrupted, str_toks, io_tok, s_tok, END=len(str_toks) - 1)


def logit_diff(logits: torch.Tensor, task: IOITask) -> float:
    """IO minus S logit at the final position. logits: [batch, pos, d_vocab].

    Positive ⇒ the model prefers the indirect object (IOI behavior present). One
    scalar you watch move as you poke at the internals."""
    last = logits[0, -1]
    return (last[task.io_tok] - last[task.s_tok]).item()


def make_ioi_metric(clean_ld: float, corrupted_ld: float):
    """Return a normalized metric: 0 at the corrupted baseline, 1 at the clean one.

    Normalizing makes patch results read as "% of the behavior restored" rather
    than raw logits, so heatmaps are comparable across hook points."""
    span = clean_ld - corrupted_ld

    def ioi_metric(logits: torch.Tensor, task: IOITask) -> float:
        return (logit_diff(logits, task) - corrupted_ld) / span

    return ioi_metric
