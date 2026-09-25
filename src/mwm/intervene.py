"""Phase 4: causal intervention (activation patching / steering).

Probing is correlational.  Here we add a concept direction to the residual
stream at a chosen layer and token position, and ask whether the model's own
downstream behaviour changes in the predicted direction.

Two read-outs, reported separately because they support different claims:

behavioural
    Append a cue sentence and compare the model's log-probability of the
    competing state words ("matte" / "dewy" / "satin").  This is the strong
    test: it asks whether the representation is used by the computation that
    produces text.  It is only interpretable if the *unintervened* model
    already reads the state out above chance, so the baseline is always
    reported alongside.

probe read-out at a later layer
    Whether the downstream layers' own state code flips.  Weaker (it stays
    inside the representation) but always available, and it localises where
    the edit propagates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import models as M


def _blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The list of transformer blocks, across the architectures we use."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers",
                 "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise AttributeError("could not locate transformer blocks")


class ResidualAdd:
    """Forward hook adding `alpha * direction` to a block's output.

    `positions` are token indices; None means every position from `from_pos`
    onward (steering the whole continuation, which is what a persistent state
    edit should look like).
    """

    def __init__(self, direction: torch.Tensor, alpha: float,
                 positions: list[int] | None = None, from_pos: int = 0):
        self.direction = direction
        self.alpha = alpha
        self.positions = positions
        self.from_pos = from_pos

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        d = self.direction.to(h.dtype).to(h.device)
        if self.positions is None:
            h[:, self.from_pos:, :] = h[:, self.from_pos:, :] + self.alpha * d
        else:
            for p in self.positions:
                h[:, p, :] = h[:, p, :] + self.alpha * d
        return (h,) + output[1:] if is_tuple else h


@dataclass
class ReadoutSpec:
    """A cue whose next-token distribution reveals the model's belief."""
    cue: str                    # appended after the narrative
    options: dict[str, str]     # state value -> continuation string


FINISH_READOUT = ReadoutSpec(
    cue=" Right now the finish on {ent} can only be described as",
    options={"matte": " matte", "dewy": " dewy", "satin": " satin"},
)
COLOR_READOUT = ReadoutSpec(
    cue=" The colour currently on {ent} is best described as",
    options={"nude": " nude", "pink": " pink", "red": " red",
             "brown": " brown"},
)
COVERAGE_READOUT = ReadoutSpec(
    cue=" The amount of product currently on {ent} would be called",
    # `bare` is deliberately excluded, for the same reason FINISH_READOUT
    # excludes `none`: the null value occurs almost only early in a narrative
    # (mean position 3.65 against 7.5-7.7 for light/full), so its
    # difference-of-means direction encodes "early in the text" rather than
    # "no product". Steering towards it moved the read-out the wrong way --
    # see docs/CORRECTIONS.md #13.
    options={"light": " light", "full": " full"},
)
READOUTS = {"finish": FINISH_READOUT, "color": COLOR_READOUT,
            "coverage": COVERAGE_READOUT}

ENTITY_CUE_NAME = {"lips": "her lips", "eyes": "her eyelids",
                   "cheeks": "her cheeks", "skin": "her skin"}


@torch.no_grad()
def readout_scores(loaded: M.Loaded, narrative: str, entity: str,
                   spec: ReadoutSpec,
                   hook_layer: int | None = None,
                   hook: ResidualAdd | None = None) -> dict[str, float]:
    """Length-normalised logprob of each option, optionally under a hook."""
    prompt = narrative + spec.cue.format(ent=ENTITY_CUE_NAME[entity])
    handle = None
    if hook is not None:
        assert hook_layer is not None
        handle = _blocks(loaded.model)[hook_layer].register_forward_hook(hook)
    try:
        return {v: M.sequence_logprob(loaded, prompt, cont)
                for v, cont in spec.options.items()}
    finally:
        if handle is not None:
            handle.remove()


def first_tokens_distinct(loaded: M.Loaded, spec: ReadoutSpec) -> bool:
    firsts = [loaded.tokenizer(c).input_ids[0] for c in spec.options.values()]
    return len(set(firsts)) == len(firsts)


@torch.no_grad()
def readout_scores_fast(loaded: M.Loaded, narrative: str, entity: str,
                        spec: ReadoutSpec, hook_layer: int | None = None,
                        hook: ResidualAdd | None = None) -> dict[str, float]:
    """Forced choice over the options' first tokens -- one forward pass.

    Valid only when the options' first tokens differ (checked by
    `first_tokens_distinct`); this is the standard cheap forced-choice
    read-out and lets us afford hundreds of intervention trials on CPU.
    """
    prompt = narrative + spec.cue.format(ent=ENTITY_CUE_NAME[entity])
    ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids.to(M.device())
    handle = None
    if hook is not None:
        assert hook_layer is not None
        handle = _blocks(loaded.model)[hook_layer].register_forward_hook(hook)
    try:
        logits = loaded.model(ids).logits[0, -1].float()
    finally:
        if handle is not None:
            handle.remove()
    lp = torch.log_softmax(logits, dim=-1)
    return {v: float(lp[loaded.tokenizer(c).input_ids[0]])
            for v, c in spec.options.items()}


@torch.no_grad()
def hidden_under_hook(loaded: M.Loaded, text: str, layer: int,
                      hook_layer: int, hook: ResidualAdd | None
                      ) -> torch.Tensor:
    """Last-token hidden state at `layer`, optionally with a hook applied."""
    tok = loaded.tokenizer
    ids = tok(text, return_tensors="pt").input_ids.to(M.device())
    handle = None
    if hook is not None:
        handle = _blocks(loaded.model)[hook_layer].register_forward_hook(hook)
    try:
        out = loaded.model(ids, output_hidden_states=True)
        return out.hidden_states[layer][0, -1].float().cpu()
    finally:
        if handle is not None:
            handle.remove()


def argmax_option(scores: dict[str, float]) -> str:
    return max(scores, key=scores.get)


def logit_gap(scores: dict[str, float], target: str) -> float:
    """logP(target) - max logP(any other option). Positive = target wins."""
    others = [v for k, v in scores.items() if k != target]
    return scores[target] - max(others)


def direction_from_activations(X: np.ndarray, y: np.ndarray,
                               source_cls: int, target_cls: int
                               ) -> torch.Tensor:
    """mean(target) - mean(source): the edit that should turn one into the other."""
    d = X[y == target_cls].mean(axis=0) - X[y == source_cls].mean(axis=0)
    return torch.from_numpy(np.asarray(d, dtype=np.float32))
