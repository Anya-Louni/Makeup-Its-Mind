"""Explicit state machine for the makeup-routine world.

Four face-region entities, each carrying three attributes.  Every narrative
sentence corresponds to exactly one action; applying the action to the state
deterministically produces the ground-truth label for that sentence position.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- vocabulary
ENTITIES: tuple[str, ...] = ("lips", "eyes", "cheeks", "skin")

# Closed palette, deliberately identical across entities so that cross-entity
# confusion is measurable (a probe for lip-colour must not be readable off the
# eye-colour subspace).
COLORS: tuple[str, ...] = ("bare", "nude", "pink", "red", "brown")
FINISHES: tuple[str, ...] = ("none", "matte", "dewy", "satin")
COVERAGES: tuple[str, ...] = ("bare", "light", "full")

ATTRIBUTES: tuple[str, ...] = ("color", "finish", "coverage")
ATTR_VALUES: dict[str, tuple[str, ...]] = {
    "color": COLORS,
    "finish": FINISHES,
    "coverage": COVERAGES,
}

# Colours/finishes that can actually be *applied* (i.e. exclude the null value)
APPLICABLE_COLORS: tuple[str, ...] = ("nude", "pink", "red", "brown")
APPLICABLE_FINISHES: tuple[str, ...] = ("matte", "dewy", "satin")

BARE = {"color": "bare", "finish": "none", "coverage": "bare"}


def label_index(attribute: str, value: str) -> int:
    return ATTR_VALUES[attribute].index(value)


# -------------------------------------------------------------------- state
@dataclass
class WorldState:
    """State of every tracked face region."""

    regions: dict[str, dict[str, str]] = field(
        default_factory=lambda: {e: dict(BARE) for e in ENTITIES}
    )

    def copy(self) -> "WorldState":
        return WorldState(regions=copy.deepcopy(self.regions))

    def get(self, entity: str, attribute: str) -> str:
        return self.regions[entity][attribute]

    def flat(self, entities: tuple[str, ...] = ENTITIES,
             attributes: tuple[str, ...] = ATTRIBUTES) -> dict[str, str]:
        """Flatten to {'lips.color': 'red', ...} for label storage."""
        return {
            f"{e}.{a}": self.regions[e][a] for e in entities for a in attributes
        }


# ------------------------------------------------------------------ actions
# An action is a small record; `apply` mutates a copy of the state.
# `kind` selects the surface-realisation template family in lexicon.py.

@dataclass
class Action:
    kind: str                       # APPLY / BLOT / GLOSS / SOFTEN / BUILD /
                                    # SHEER / REMOVE / NOOP / DISTRACTOR
    entity: str | None = None
    color: str | None = None
    finish: str | None = None
    coverage: str | None = None
    # DISTRACTOR carries a colour word that is *mentioned but not applied*.
    mentioned_color: str | None = None
    mentioned_entity: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def apply_action(state: WorldState, action: Action) -> WorldState:
    """Return a new state with `action` applied. Pure function."""
    s = state.copy()
    k = action.kind

    if k in ("NOOP", "DISTRACTOR"):
        return s

    e = action.entity
    assert e is not None, f"action {k} needs an entity"
    r = s.regions[e]

    if k == "APPLY":
        r["color"] = action.color
        r["finish"] = action.finish
        r["coverage"] = action.coverage
    elif k == "BLOT":
        r["finish"] = "matte"
    elif k == "GLOSS":
        r["finish"] = "dewy"
    elif k == "SOFTEN":
        r["finish"] = "satin"
    elif k == "BUILD":
        r["coverage"] = "full"
    elif k == "SHEER":
        r["coverage"] = "light"
    elif k == "REMOVE":
        s.regions[e] = dict(BARE)
    else:
        raise ValueError(f"unknown action kind: {k}")
    return s


def legal_actions(state: WorldState, entities: tuple[str, ...]) -> list[Action]:
    """Enumerate actions that are meaningful given the current state."""
    out: list[Action] = []
    for e in entities:
        r = state.regions[e]
        painted = r["coverage"] != "bare"
        # APPLY is always legal (re-application = override)
        for c in APPLICABLE_COLORS:
            for f in APPLICABLE_FINISHES:
                for cov in ("light", "full"):
                    out.append(Action("APPLY", e, color=c, finish=f, coverage=cov))
        if painted:
            if r["finish"] != "matte":
                out.append(Action("BLOT", e))
            if r["finish"] != "dewy":
                out.append(Action("GLOSS", e))
            if r["finish"] != "satin":
                out.append(Action("SOFTEN", e))
            if r["coverage"] == "light":
                out.append(Action("BUILD", e))
            if r["coverage"] == "full":
                out.append(Action("SHEER", e))
            out.append(Action("REMOVE", e))
    return out
