"""Activation extraction over a generated dataset.

Produces, for every (narrative, sentence-position) pair:

  X.npy   float16 (N, n_layers, hidden)   residual stream at the last token
  Xmean.npy float16 (N, n_layers, hidden) residual stream mean-pooled over the
                                          whole prefix (fair comparison with
                                          the pooled non-contextual baseline)
  E.npy   float16 (N, hidden)             non-contextual embedding baseline
  meta.jsonl                              labels, split, condition, flags,
                                          and the prefix text (for the
                                          bag-of-ngrams baseline)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import models as M


def extract_dataset(records: list[dict], model_name: str, out_dir: str | Path,
                    layers: list[int] | None = None,
                    progress_every: int = 100,
                    dtype: torch.dtype = torch.float32) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_positions = sum(len(r["sentences"]) for r in records)

    with M.loaded_model(model_name, dtype) as obj:
        sel = layers if layers is not None else list(range(obj.n_layers + 1))
        X = np.lib.format.open_memmap(
            out_dir / "X.npy", mode="w+", dtype=np.float16,
            shape=(n_positions, len(sel), obj.hidden_size))
        Xm = np.lib.format.open_memmap(
            out_dir / "Xmean.npy", mode="w+", dtype=np.float16,
            shape=(n_positions, len(sel), obj.hidden_size))
        E = np.lib.format.open_memmap(
            out_dir / "E.npy", mode="w+", dtype=np.float16,
            shape=(n_positions, obj.hidden_size))

        row = 0
        with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as fh:
            for n_done, rec in enumerate(records):
                sents = rec["sentences"]
                hs_last, hs_mean = M.sentence_states(obj, sents, sel)
                hs = hs_last.numpy()
                em = M.embedding_baseline(obj, sents).numpy()
                X[row:row + len(sents)] = hs.astype(np.float16)
                Xm[row:row + len(sents)] = hs_mean.numpy().astype(np.float16)
                E[row:row + len(sents)] = em.astype(np.float16)
                prefix = ""
                for i, s in enumerate(sents):
                    prefix = s if not prefix else prefix + " " + s
                    fh.write(json.dumps({
                        "row": row + i,
                        "narrative_id": rec["id"],
                        "split": rec["split"],
                        "condition": rec["condition"],
                        "position": i,
                        "n_sentences": len(sents),
                        "action_kind": rec["action_kinds"][i],
                        "labels": rec["labels"][i],
                        "order_sensitive": rec["order_sensitive"][i],
                        "text": prefix,
                    }) + "\n")
                row += len(sents)
                if progress_every and (n_done + 1) % progress_every == 0:
                    print(f"  {n_done + 1}/{len(records)} narratives",
                          flush=True)

        X.flush()
        Xm.flush()
        E.flush()
        info = {
            "model": model_name,
            "layers": sel,
            "hidden_size": obj.hidden_size,
            "n_layers_total": obj.n_layers,
            "n_positions": n_positions,
            "n_narratives": len(records),
        }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2),
                                       encoding="utf-8")
    return info


def load_pooled(out_dir: str | Path):
    """Mean-pooled hidden states, if this extraction stored them."""
    p = Path(out_dir) / "Xmean.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def load_extracted(out_dir: str | Path):
    out_dir = Path(out_dir)
    X = np.load(out_dir / "X.npy", mmap_mode="r")
    E = np.load(out_dir / "E.npy", mmap_mode="r")
    meta = [json.loads(l) for l in
            (out_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    info = json.loads((out_dir / "info.json").read_text(encoding="utf-8"))
    return X, E, meta, info
