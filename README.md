# interp-lab

A mechanistic interpretability sandbox. Learning-by-doing, notebook-driven —
not a product, not a library.

The arc:
1. **IOI** — replicate the Indirect Object Identification circuit on GPT-2 small
   (Wang et al., 2022). Activation patching, attention-head attribution, the
   name-mover / S-inhibition / induction story.
2. **SAE** — train a small sparse autoencoder on a residual-stream layer and
   poke at the features it learns.

Target model is **GPT-2 small** (124M) throughout. It's tiny — hardware is a
non-issue here. This is interpretability (full activation access, raw weights),
not inference serving.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab          # or: jupyter notebook
```

CUDA is used if present (the lab box is an RTX 3090 / Ampere; GPT-2 small fits
in a sliver of 24GB). Falls back to CPU — still fine for a 124M model, just
slower. First run downloads the GPT-2 weights into the HuggingFace cache.

### Quick sanity check

`notebooks/00_ioi_baseline.ipynb` loads the model, runs the canonical IOI
prompt, and confirms the behavior:

```
prompt:  "When John and Mary went to the store, John gave a drink to"
top token: ' Mary'
logit_diff (IO − S): ≈ +3.2     → the IOI behavior is present
```

If that logit diff is comfortably positive, the circuit you're about to take
apart is actually there.

## Layout

```
notebooks/    the work — one notebook per investigation, numbered in order
requirements.txt
LICENSE       MIT
```

Flat and notebook-first on purpose. No premature abstraction — if something
earns being pulled into a shared module later, it will. Until then it lives in
the notebook that needs it.

## Stack

- [transformer-lens](https://github.com/TransformerLensOrg/TransformerLens) —
  `HookedTransformer`, `run_with_cache`, activation patching. The core tool.
- [torch](https://pytorch.org/) — CUDA.
- jupyter.
- [sae-lens](https://github.com/jbloomAus/SAELens) — added in the SAE phase.

## References

- Wang et al., *Interpretability in the Wild: a Circuit for IOI in GPT-2 small*
  (2022) — the circuit being replicated.
- Nanda & Bloom, the transformer-lens / ARENA materials — method.

## License

MIT — see [LICENSE](LICENSE).
