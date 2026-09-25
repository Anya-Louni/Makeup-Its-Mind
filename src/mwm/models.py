"""Model loading, hidden-state extraction and lightweight scoring helpers.

Everything here is CPU-safe: the reference machine for this project has no
CUDA device and ~15 GB of system RAM, so models are loaded one at a time and
explicitly released.
"""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CANDIDATES: tuple[str, ...] = (
    "gpt2",                       # 124M
    "EleutherAI/pythia-410m",     # 410M
    "HuggingFaceTB/SmolLM2-360M",  # 360M, modern data mix
    "Qwen/Qwen2.5-0.5B",          # 494M
    "gpt2-medium",                # 355M
    "meta-llama/Llama-3.2-1B",    # 1.2B  (gated on the Hub)
    "google/gemma-2-2b",          # 2.6B  (gated; likely OOM on 15 GB CPU)
)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Loaded:
    name: str
    model: torch.nn.Module
    tokenizer: object
    n_layers: int
    hidden_size: int
    n_params: int
    load_seconds: float


def load(name: str, dtype: torch.dtype = torch.float32) -> Loaded:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.eval()
    model.to(device())
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    return Loaded(
        name=name,
        model=model,
        tokenizer=tok,
        n_layers=int(n_layers),
        hidden_size=int(hidden),
        n_params=sum(p.numel() for p in model.parameters()),
        load_seconds=time.time() - t0,
    )


def release(loaded: Loaded | None) -> None:
    if loaded is None:
        return
    del loaded.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def loaded_model(name: str, dtype: torch.dtype = torch.float32):
    obj = None
    try:
        obj = load(name, dtype)
        yield obj
    finally:
        release(obj)


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def sequence_logprob(loaded: Loaded, prefix: str, continuation: str) -> float:
    """Total log P(continuation | prefix), length-normalised per token."""
    tok, model = loaded.tokenizer, loaded.model
    pre_ids = tok(prefix, return_tensors="pt").input_ids
    full_ids = tok(prefix + continuation, return_tensors="pt").input_ids
    full_ids = full_ids.to(device())
    n_pre = pre_ids.shape[1]
    n_cont = full_ids.shape[1] - n_pre
    if n_cont <= 0:
        return float("nan")
    logits = model(full_ids).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for i in range(n_pre, full_ids.shape[1]):
        total += logprobs[i - 1, full_ids[0, i]].item()
    return total / n_cont


# ------------------------------------------------- hidden-state extraction
@torch.no_grad()
def sentence_hidden_states(loaded: Loaded, sentences: list[str],
                           layers: list[int] | None = None) -> torch.Tensor:
    """Residual-stream state at the final token of each sentence.

    The narrative is encoded once, incrementally: sentence boundaries are
    located by tokenising cumulative prefixes, so position i is the model's
    representation "as of" having read sentences 0..i.

    Returns a tensor of shape (n_sentences, n_selected_layers, hidden_size).
    """
    tok, model = loaded.tokenizer, loaded.model
    text = ""
    boundaries: list[int] = []
    for s in sentences:
        text = s if not text else text + " " + s
        boundaries.append(len(tok(text).input_ids) - 1)

    ids = tok(text, return_tensors="pt").input_ids.to(device())
    out = model(ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0]     # (L+1, T, H)
    if layers is None:
        layers = list(range(hs.shape[0]))
    hs = hs[layers]                                       # (L', T, H)
    idx = torch.tensor(boundaries, device=hs.device)
    picked = hs.index_select(1, idx)                      # (L', n_sent, H)
    return picked.permute(1, 0, 2).contiguous().float().cpu()


@torch.no_grad()
def sentence_states(loaded: Loaded, sentences: list[str],
                    layers: list[int] | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Both read-out conventions from a single forward pass.

    Returns (last_token, mean_pooled), each (n_sentences, n_layers, hidden).

    `last_token` is the standard convention: the residual stream at the final
    token of each sentence.  `mean_pooled` averages the residual stream over
    every token of the prefix so far.

    Both are needed for a fair comparison: the non-contextual baseline pools
    over the whole prefix, so a last-token hidden state is being asked to beat
    it without the same pooling.  Reporting only one of these would either
    flatter or unfairly penalise the contextual representation.
    """
    tok, model = loaded.tokenizer, loaded.model
    text = ""
    boundaries: list[int] = []
    for s in sentences:
        text = s if not text else text + " " + s
        boundaries.append(len(tok(text).input_ids) - 1)

    ids = tok(text, return_tensors="pt").input_ids.to(device())
    out = model(ids, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0]      # (L+1, T, H)
    if layers is None:
        layers = list(range(hs.shape[0]))
    hs = hs[layers].float()                               # (L', T, H)

    idx = torch.tensor(boundaries, device=hs.device)
    last = hs.index_select(1, idx).permute(1, 0, 2).contiguous()
    csum = torch.cumsum(hs, dim=1)
    counts = (idx + 1).to(hs.dtype).view(1, -1, 1)
    mean = (csum.index_select(1, idx) / counts).permute(1, 0, 2).contiguous()
    return last.cpu(), mean.cpu()


@torch.no_grad()
def embedding_baseline(loaded: Loaded, sentences: list[str]) -> torch.Tensor:
    """Non-contextual control: mean of input embeddings over the prefix.

    This is layer 0 *without* any attention, i.e. a pure bag-of-tokens vector
    in the model's own embedding space.  Shape (n_sentences, hidden_size).
    """
    tok = loaded.tokenizer
    emb = loaded.model.get_input_embeddings()
    text = ""
    boundaries: list[int] = []
    for s in sentences:
        text = s if not text else text + " " + s
        boundaries.append(len(tok(text).input_ids) - 1)

    ids = tok(text, return_tensors="pt").input_ids.to(device())
    e = emb(ids)[0].float()                       # (T, H)
    csum = torch.cumsum(e, dim=0)
    idx = torch.tensor(boundaries, device=e.device)
    counts = (idx + 1).unsqueeze(1).to(e.dtype)
    return (csum.index_select(0, idx) / counts).cpu()
