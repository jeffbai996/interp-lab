# interp-lab

A mechanistic interpretability sandbox. Notebook-driven, learning-by-doing —
not a product, not a library. The goal is to *reverse-engineer* small pieces of
a language model by hand, not to use one.

The arc:

1. **IOI** — replicate the Indirect Object Identification circuit on GPT-2 small
   (Wang et al., 2022): activation patching, attention-head attribution, and the
   name-mover / S-inhibition / induction-head story.
2. **SAE** — train a small sparse autoencoder on a residual-stream layer and
   inspect the features it learns.

Target model is **GPT-2 small** (124M params, 12 layers, 12 heads, d_model 768)
throughout. It's tiny on purpose: small enough to hold the whole thing in your
head, big enough to do real circuits. This is interpretability — full activation
access, raw fp32/fp16 weights — not inference serving, so there's no
quantization or serving tooling here by design.

## What is this actually about?

A transformer maps tokens → next-token logits, and the *how* is buried in
millions of weights. Mechanistic interpretability is the project of recovering
human-understandable algorithms ("circuits") from those weights.

The trick that makes it tractable: you don't read weights directly. You run the
model with full instrumentation, then **intervene** on its internal activations
and watch the output change. Whatever you can change to break (or restore) a
behavior *is*, causally, the mechanism behind it.

**IOI** is the canonical first circuit because the behavior is crisp and the
metric is a single number. Given

> "When John and Mary went to the store, John gave a drink to ___"

a competent model continues with **" Mary"** — the indirect object, mentioned
once — and not **" John"**, the subject, mentioned twice. The metric is the
**logit difference**:

```
logit_diff = logit(" Mary") − logit(" John")
```

Positive ⇒ the behavior is present. One scalar you can watch move as you poke at
the internals.

## The core technique: activation patching

This is ~90% of the IOI work, so it's worth stating up front.

You run two forward passes:

- **clean** — the prompt above; the model wants `" Mary"`. Logit diff is high.
- **corrupted** — one thing changed (e.g. swap a name so the answer flips, or
  replace a name with a random token). Logit diff collapses.

Now the move: run the **corrupted** prompt, but at one chosen hook point — say
the output of head 9.9 at the final token — **overwrite** that activation with
the value it had in the *clean* run. Everything else stays corrupted. Read the
logit diff.

- If splicing that one clean activation back **restores** the answer, that
  activation was *causally carrying the signal*.
- If nothing changes, it wasn't.

Sweep over every `(layer, position)` → a heatmap of *where* the behavior lives.
Sweep over every `(layer, head)` → *which heads* implement it. The whole IOI
paper is this one primitive applied at increasing resolution.

Why it's **causal**, not correlational: probing asks "does this activation
correlate with the output?" Patching asks "if I *set* this activation to its
clean value, does the behavior come back?" — an intervention, which is what makes
the resulting circuit claims actually hold.

It works cleanly because of the **residual stream**: every layer reads from and
writes to a shared `[position, d_model]` stream *additively* (the residual
connections). Each attention head's contribution is just a vector added in, so
you can surgically swap one head's output without disturbing the rest. That
linearity is the whole reason patching is surgical instead of chaotic.

## What you'll find (GPT-2 small, IOI)

The circuit decomposes into a few head families that compose:

- **Duplicate-token / induction heads** (early layers) — detect that `"John"`
  appeared twice.
- **S-inhibition heads** (middle layers) — write a signal that suppresses the
  duplicated *subject*, so the answer-writers don't copy it.
- **Name-mover heads** (late, ~L9–L10) — attend from the final position to the
  *indirect object* and copy it into the output. The actual answer-writers.

Roughly: detect the duplicate → inhibit the subject → move the remaining name.

## Results so far

The full IOI circuit in GPT-2 small, mapped by hand across the notebooks. Clean
logit diff ≈ **+3.17**, corrupted ≈ **−3.54** (the metric the whole lab moves).

| stage | heads | found in |
|---|---|---|
| previous-token | 2.2, 4.11 | `03` |
| duplicate-token | 0.1, 0.10, 3.0 | `03` |
| induction | 5.5, 5.8, 5.9, 6.9 | `03` |
| S-inhibition | 7.3, 7.9, 8.6, 8.10 | `02` |
| name movers | 9.6, 9.9, 10.0 | `02` |
| negative movers | 10.7, 11.10 | `02` |
| backup movers | 9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.6, 11.9 | `02` |

Read it as a pipeline: **detect the duplicate** (previous-token feeds induction;
induction and duplicate-token both write "S2 is the repeat") **→ inhibit the
subject** (S-inhibition reads S2 and steers the movers off it) **→ move the name**
(name movers copy the indirect object). Every head above is a heatmap cell you
produce yourself, then confirm with ablation. Notebook **04** then sets circuits
aside for the SAE feature basis.

Shared setup (model loader, the IOI task + metric, the heatmap) lives in
`interp_lab/` from `03` onward; `00–02` predate it and inline their own copies.

## Setup

```bash
python3 -m venv venv            # or: uv venv && source .venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
jupyter lab                     # or: jupyter notebook
```

A CUDA GPU is used automatically if present (any modern card is wild overkill —
GPT-2 small + its activations fit in a couple hundred MB). Falls back to CPU,
which is still fine for a 124M model, just slower. The first model load downloads
the GPT-2 weights into the HuggingFace cache (`~/.cache/huggingface`).

### Quick sanity check

Open `notebooks/00_ioi_baseline.ipynb` and run it top to bottom. It loads the
model, runs the canonical prompt plus a few role-swapped variants, and confirms:

```
device: cuda
top predicted token: ' Mary'
logit_diff (IO − S): ≈ +3.2          → IOI behavior present
+3.17  IO=Mary  S=John     +2.48  IO=John  S=Mary
+3.79  IO=Alice S=Bob      +2.73  IO=Bob   S=Alice
```

All four role-swaps positive ⇒ the circuit tracks the indirect-object *role*, not
the literal string `"Mary"`. If that holds, the circuit you're about to take
apart is genuinely there.

## The core API (this is most of what you'll use)

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")   # GPT-2 small + hooks

logits = model(prompt)                              # ordinary forward pass
logits, cache = model.run_with_cache(prompt)        # ...and keep every activation

cache["resid_post", 9]   # residual stream after layer 9   [batch, pos, d_model]
cache["pattern", 9]      # attention pattern, layer 9       [batch, head, q, k]
cache["z", 9]            # per-head output, layer 9         [batch, pos, head, d_head]

# patch an activation mid-forward-pass:
def hook(act, hook):
    act[:, pos, head] = clean_cache["z", 9][:, pos, head]   # splice in clean value
    return act
patched = model.run_with_hooks(prompt, fwd_hooks=[("blocks.9.attn.hook_z", hook)])
```

`run_with_cache` records the tape; `run_with_hooks` splices and replays it.
Everything in IOI is built from those two.

## Working through it (and actually learning, not just running cells)

Notebooks are numbered as a curriculum:

- **00 — baseline** *(included)*: confirm the behavior exists and the metric
  works. Always pin your metric before you start cutting.
- **01 — activation patching**: localize the circuit by `(layer, position)`.
- **02 — head attribution**: DLA vs. patching, attention patterns, and the
  S-inhibition heads — the *back* of the circuit (the answer-writers and what
  steers them).
- **03 — duplicate & induction heads**: the *front* of the circuit — how the
  model detects that the subject is the *repeated* name (previous-token,
  duplicate-token, and induction heads), closing the IOI loop.
- **04 — SAEs**: a different question — not "what does this circuit do" but "what
  features does the residual stream encode in a human-readable basis." Load a
  pretrained sparse autoencoder, inspect features, and tie them back to the circuit.

The way to actually internalize this, rather than nodding along:

- **Predict before you run.** Before each patch, write down what you expect the
  logit diff to do. The gap between your prediction and the result is the lesson.
- **Break things on purpose.** Change the names, lengthen the sentence, add a
  third name. A circuit you can break is one you understand.
- **Ablate, don't only patch.** Zero a head out entirely; if the behavior
  survives, that head wasn't load-bearing — a useful way to falsify your own
  hypotheses.
- Keep a "expected vs. observed" log in a markdown cell at the top of each
  notebook. That diff is the curriculum.

## Layout

```
notebooks/         one notebook per investigation, numbered in order
interp_lab/        shared helpers (model loader, IOI task + metric, heatmap)
requirements.txt   pinned environment
LICENSE            MIT
```

Flat and notebook-first on purpose. No premature abstraction — the `interp_lab/`
module only appeared once the IOI setup, metric, and heatmap had been copy-pasted
across three notebooks, which is the bar: if something earns being pulled into a
shared module, it does; until then it lives in the notebook that needs it. (00–02
still inline their copies — they were written to stand alone.)

## Stack

- [transformer-lens](https://github.com/TransformerLensOrg/TransformerLens) —
  `HookedTransformer`, `run_with_cache`, hooks/patching. The core tool.
- [torch](https://pytorch.org/) — CUDA if available.
- jupyter.
- [sae-lens](https://github.com/jbloomAus/SAELens) — added in the SAE phase.

Versions are pinned in `requirements.txt`.

## Learning resources

- **[ARENA](https://arena3-chapter1-transformer-interp.streamlit.app/)** —
  the IOI + SAE curriculum this lab mirrors, with full solutions and exercises.
  The best hands-on starting point.
- **Neel Nanda** — [*A Comprehensive Mechanistic Interpretability Explainer &
  Glossary*](https://www.neelnanda.io/mechanistic-interpretability/glossary) and
  *200 Concrete Open Problems in Interpretability* for vocabulary and directions.
- **Wang et al. (2022)** — [*Interpretability in the Wild: a Circuit for IOI in
  GPT-2 small*](https://arxiv.org/abs/2211.00593). The paper being replicated;
  best read *after* notebook 02, when the heads it names will mean something.
- **transformer-lens** —
  [docs](https://transformerlensorg.github.io/TransformerLens/) and its `main_demo`
  notebook for the full hook/cache API.

## License

MIT — see [LICENSE](LICENSE).
