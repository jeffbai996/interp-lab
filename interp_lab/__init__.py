"""interp-lab shared helpers.

Pulled out of the notebooks once the IOI setup, the logit-diff metric, and the
heatmap had been copy-pasted one too many times — the README's "if something
earns being pulled into a shared module, it will" rule, finally cashed in.

Notebooks 00–02 predate this and keep their inline copies (they stand alone on
purpose); 03 onward import from here. Notebooks add the repo root to the path:

    import sys; sys.path.insert(0, "..")
    from interp_lab import load_model, ioi_task, logit_diff, make_ioi_metric, heatmap
"""
from .ioi import (
    CLEAN_PROMPT,
    CORRUPTED_PROMPT,
    IOITask,
    ioi_task,
    logit_diff,
    make_ioi_metric,
)
from .model import load_model
from .viz import heatmap

__all__ = [
    "load_model",
    "CLEAN_PROMPT",
    "CORRUPTED_PROMPT",
    "IOITask",
    "ioi_task",
    "logit_diff",
    "make_ioi_metric",
    "heatmap",
]
