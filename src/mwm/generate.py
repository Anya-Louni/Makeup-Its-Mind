"""Narrative generation from the explicit state machine.

Every narrative is a list of sentences; every sentence is the surface form of
exactly one action; the label at position i is the full world state after
executing actions[0..i].  Labels are therefore free and exact.

Split discipline: `split` selects a disjoint sixth/fifth/fifth of *every*
phrase bank and *every* sentence skeleton, so a test narrative contains no
wording that appeared in training.
"""
from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import lexicon as lx
from .state import (
    ATTRIBUTES,
    ENTITIES,
    Action,
    WorldState,
    apply_action,
    legal_actions,
)

APPLICABLE_COLORS = ("nude", "pink", "red", "brown")
APPLICABLE_FINISHES = ("matte", "dewy", "satin")

KIND_WEIGHTS = {
    "APPLY": 0.30,
    "BLOT": 0.07,
    "GLOSS": 0.07,
    "SOFTEN": 0.07,
    "BUILD": 0.08,
    "SHEER": 0.06,
    "REMOVE": 0.04,
    "NOOP": 0.10,
    "DISTRACTOR": 0.21,
}


@dataclass
class GenConfig:
    entities: tuple[str, ...] = ENTITIES
    attributes: tuple[str, ...] = ATTRIBUTES
    min_sentences: int = 8
    max_sentences: int = 14
    # guarantee at least this many entities get re-applied (override) so the
    # dataset always contains positions where "first mention" is the wrong answer
    min_overrides: int = 1
    condition: str = "natural"          # natural | shuffled

    def scale_to_entities(self) -> "GenConfig":
        n = len(self.entities)
        return GenConfig(
            entities=self.entities,
            attributes=self.attributes,
            min_sentences=max(5, 3 * n - 2),
            max_sentences=max(8, 4 * n),
            min_overrides=self.min_overrides,
            condition=self.condition,
        )


class Renderer:
    """Split-restricted surface realiser."""

    def __init__(self, split: str, rng: random.Random):
        self.rng = rng
        self.ent = {k: lx.bank_for_split(v, split)
                    for k, v in lx.ENTITY_PHRASES.items()}
        self.col = {k: lx.bank_for_split(v, split)
                    for k, v in lx.COLOR_PHRASES.items()}
        self.fin = {k: lx.bank_for_split(v, split)
                    for k, v in lx.FINISH_PHRASES.items()}
        self.cov = {k: lx.bank_for_split(v, split)
                    for k, v in lx.COVERAGE_PHRASES.items()}
        self.tpl = {k: lx.bank_for_split(v, split)
                    for k, v in lx.TEMPLATES.items()}

    def render(self, action: Action) -> str:
        r = self.rng
        vals: dict[str, str] = {}
        if action.entity is not None:
            vals["ent"] = r.choice(self.ent[action.entity])
        if action.color is not None:
            vals["col"] = r.choice(self.col[action.color])
        if action.finish is not None:
            vals["fin"] = r.choice(self.fin[action.finish])
        if action.coverage is not None and action.coverage in self.cov:
            vals["cov"] = r.choice(self.cov[action.coverage])
        if action.mentioned_color is not None:
            vals["mcol"] = r.choice(self.col[action.mentioned_color])
        if action.mentioned_entity is not None:
            vals["ment"] = r.choice(self.ent[action.mentioned_entity])
        return lx.fill(r.choice(self.tpl[action.kind]), vals)


def _sample_action(state: WorldState, cfg: GenConfig, rng: random.Random,
                   force_apply_on: str | None) -> Action:
    if force_apply_on is not None:
        return Action("APPLY", force_apply_on,
                      color=rng.choice(APPLICABLE_COLORS),
                      finish=rng.choice(APPLICABLE_FINISHES),
                      coverage=rng.choice(("light", "full")))

    legal = legal_actions(state, cfg.entities)
    by_kind: dict[str, list[Action]] = {}
    for a in legal:
        by_kind.setdefault(a.kind, []).append(a)
    by_kind["NOOP"] = [Action("NOOP")]
    by_kind["DISTRACTOR"] = [
        Action("DISTRACTOR",
               mentioned_color=rng.choice(APPLICABLE_COLORS),
               mentioned_entity=rng.choice(cfg.entities))
    ]

    kinds = [k for k in by_kind if k in KIND_WEIGHTS]
    weights = [KIND_WEIGHTS[k] for k in kinds]
    kind = rng.choices(kinds, weights=weights, k=1)[0]
    return rng.choice(by_kind[kind])


def generate_one(idx: int, split: str, cfg: GenConfig,
                 rng: random.Random) -> dict:
    n = rng.randint(cfg.min_sentences, cfg.max_sentences)
    ents = list(cfg.entities)

    # Plan which steps must be a first APPLY so that every entity gets painted.
    first_apply_steps = sorted(rng.sample(range(n), len(ents)))
    rng.shuffle(ents)
    forced = dict(zip(first_apply_steps, ents))

    state = WorldState()
    actions: list[Action] = []
    states: list[WorldState] = []
    for step in range(n):
        a = _sample_action(state, cfg, rng, forced.get(step))
        state = apply_action(state, a)
        actions.append(a)
        states.append(state)

    # Guarantee at least `min_overrides` genuine overrides: an entity whose
    # value for some attribute changed after it was first set.
    overrides = _count_overrides(states, cfg)
    if overrides < cfg.min_overrides:
        e = rng.choice(list(cfg.entities))
        cur = state.regions[e]
        new_color = rng.choice([c for c in APPLICABLE_COLORS
                                if c != cur["color"]])
        a = Action("APPLY", e, color=new_color,
                   finish=rng.choice(APPLICABLE_FINISHES),
                   coverage=rng.choice(("light", "full")))
        state = apply_action(state, a)
        actions.append(a)
        states.append(state)

    order = list(range(len(actions)))
    if cfg.condition == "shuffled":
        rng.shuffle(order)
        actions = [actions[i] for i in order]
        # re-execute so labels stay exact for the presented order
        s = WorldState()
        states = []
        for a in actions:
            s = apply_action(s, a)
            states.append(s)

    renderer = Renderer(split, rng)
    sentences = [renderer.render(a) for a in actions]

    # Per-position analysis flags.
    seen_first: dict[str, str] = {}
    labels, flags = [], []
    for i, s in enumerate(states):
        flat = s.flat(cfg.entities, cfg.attributes)
        labels.append(flat)
        f = {}
        for key, val in flat.items():
            if key not in seen_first and val not in ("bare", "none"):
                seen_first[key] = val
            # order_sensitive: the correct answer is NOT the first value ever
            # asserted for this entity-attribute
            f[key] = bool(key in seen_first and seen_first[key] != val)
        flags.append(f)

    return {
        "id": f"{split}-{cfg.condition}-{idx:06d}",
        "split": split,
        "condition": cfg.condition,
        "entities": list(cfg.entities),
        "attributes": list(cfg.attributes),
        "sentences": sentences,
        "action_kinds": [a.kind for a in actions],
        "actions": [a.as_dict() for a in actions],
        "labels": labels,
        "order_sensitive": flags,
    }


def _count_overrides(states: list[WorldState], cfg: GenConfig) -> int:
    seen: dict[str, str] = {}
    n = 0
    for s in states:
        for key, val in s.flat(cfg.entities, cfg.attributes).items():
            if key in seen and seen[key] != val and val not in ("bare", "none"):
                n += 1
            if val not in ("bare", "none"):
                seen[key] = val
    return n


def generate_dataset(n_per_split: dict[str, int], cfg: GenConfig,
                     seed: int = 0) -> list[dict]:
    out = []
    for split, n in n_per_split.items():
        # zlib.crc32 rather than hash(): Python randomises string hashing per
        # process, so hash() made the "seed" argument meaningless across runs.
        key = f"{seed}|{split}|{cfg.condition}".encode()
        rng = random.Random(zlib.crc32(key))
        for i in range(n):
            out.append(generate_one(i, split, cfg, rng))
    return out


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
