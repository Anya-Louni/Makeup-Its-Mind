"""Probing: linear / MLP probes, baselines, control tasks and MDL.

Everything takes the same (features, labels, group) interface so the same
evaluation harness runs over hidden states, the non-contextual embedding
baseline and the bag-of-ngrams baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ helpers
def target_vector(meta: list[dict], key: str) -> np.ndarray:
    return np.array([m["labels"][key] for m in meta])


def split_mask(meta: list[dict], split: str, condition: str | None = None,
               min_position: int = 0) -> np.ndarray:
    return np.array([
        m["split"] == split
        and (condition is None or m["condition"] == condition)
        and m["position"] >= min_position
        for m in meta
    ])


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(y.tolist()))
    lut = {c: i for i, c in enumerate(classes)}
    return np.array([lut[v] for v in y]), classes


@dataclass
class ProbeResult:
    accuracy: float
    macro_f1: float
    majority: float
    n_train: int
    n_test: int
    n_classes: int
    train_accuracy: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp == 0:
            f1s.append(0.0)
        else:
            p, r = tp / (tp + fp), tp / (tp + fn)
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s))


def make_probe(kind: str, hidden: int = 64, seed: int = 0,
               max_iter: int = 1000, C: float = 1.0):
    if kind == "linear":
        return LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=max_iter,
                             random_state=seed, early_stopping=True,
                             n_iter_no_change=10)
    raise ValueError(kind)


def fit_eval(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
             kind: str = "linear", hidden: int = 64, seed: int = 0,
             standardize: bool = True) -> ProbeResult:
    if standardize:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    k = int(max(ytr.max(), yte.max())) + 1
    clf = make_probe(kind, hidden, seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    counts = np.bincount(ytr, minlength=k)
    maj = float(np.mean(yte == counts.argmax()))
    return ProbeResult(
        accuracy=float(np.mean(pred == yte)),
        macro_f1=_macro_f1(yte, pred, k),
        majority=maj,
        n_train=len(ytr), n_test=len(yte), n_classes=k,
        train_accuracy=float(np.mean(clf.predict(Xtr) == ytr)),
    )


# ------------------------------------------------------------- control task
def control_labels(meta: list[dict], y: np.ndarray, mask: np.ndarray,
                   seed: int = 0) -> np.ndarray:
    """Hewitt & Liang style control task.

    Each narrative is assigned a random label drawn from the empirical label
    marginal, held constant across its positions.  The label distribution and
    the grouping structure match the real task; only the mapping from
    representation to label is arbitrary.

    Important caveat, and the reason `train_accuracy` is reported alongside:
    our train and test splits share no narratives (by design -- they share no
    wordings either), so a probe cannot carry a memorised narrative-level
    mapping across the split.  Control-task *test* accuracy is therefore at
    chance by construction, and the raw H&L selectivity gap flatters the
    result.  The capacity claim rests on control-task *train* accuracy (how
    much arbitrary structure the probe can absorb at all), on the MDL numbers,
    and on the capacity ablation -- not on the test-side gap alone.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y[mask], return_counts=True)
    p = counts / counts.sum()
    lut: dict[str, int] = {}
    out = np.empty(len(meta), dtype=int)
    for i, m in enumerate(meta):
        nid = m["narrative_id"]
        if nid not in lut:
            lut[nid] = int(rng.choice(len(classes), p=p))
        out[i] = lut[nid]
    return out


# --------------------------------------------------------------------- MDL
def mdl_online_codelength(X: np.ndarray, y: np.ndarray, n_classes: int,
                          fractions: tuple[float, ...] = (
                              0.002, 0.004, 0.008, 0.016, 0.032, 0.0625,
                              0.125, 0.25, 0.5, 1.0),
                          kind: str = "linear", hidden: int = 64,
                          seed: int = 0, C: float = 1.0) -> dict:
    """Prequential (online) code length, Voita & Titov 2020.

    Returns codelength in kbits and compression relative to the uniform code
    (n * log2 K).  Lower codelength = the representation makes the labels
    genuinely cheaper to transmit, which -- unlike raw accuracy -- accounts for
    the cost of the probe itself.

    `C` matters more here than it does for accuracy.  The first blocks contain
    a handful of examples against 768 features, and an unregularised logistic
    regression is confidently wrong on them; prequential coding charges
    -log2 p for exactly those confident errors, so a probe that is merely
    over-confident can score *worse* than the uniform code while still being
    the most accurate probe available.  Report a regularised variant alongside
    the default before concluding that a representation fails to compress.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    n = len(y)

    blocks = [max(8, int(round(f * n))) for f in fractions]
    blocks = sorted(set(min(b, n) for b in blocks))
    if blocks[-1] != n:
        blocks.append(n)

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    # first block transmitted with the uniform code
    t0 = blocks[0]
    codelength = t0 * math.log2(n_classes)
    for a, b in zip(blocks[:-1], blocks[1:]):
        clf = make_probe(kind, hidden, seed, C=C)
        try:
            clf.fit(Xs[:a], y[:a])
        except ValueError:                    # single class in the prefix
            codelength += (b - a) * math.log2(n_classes)
            continue
        proba = clf.predict_proba(Xs[a:b])
        known = list(clf.classes_)
        p = np.full(b - a, 1e-9)
        for j, yi in enumerate(y[a:b]):
            if yi in known:
                p[j] = max(proba[j, known.index(yi)], 1e-9)
        codelength += float(-np.sum(np.log2(p)))

    uniform = n * math.log2(n_classes)
    return {
        "C": C,
        "codelength_kbits": codelength / 1024,
        "uniform_kbits": uniform / 1024,
        "compression": uniform / codelength,
        "n": n,
        "n_classes": n_classes,
    }


# ------------------------------------------------------- ngram text baseline
def ngram_baseline(train_texts: list[str], ytr: np.ndarray,
                   test_texts: list[str], yte: np.ndarray,
                   max_features: int = 30000) -> ProbeResult:
    """Bag-of-ngrams logistic regression on the raw prefix text.

    This is the strongest purely-lexical competitor: if a TF-IDF model over
    the same text does as well as the hidden states, nothing about contextual
    representation has been demonstrated.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                          sublinear_tf=True)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    k = int(max(ytr.max(), yte.max())) + 1
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    counts = np.bincount(ytr, minlength=k)
    return ProbeResult(
        accuracy=float(np.mean(pred == yte)),
        macro_f1=_macro_f1(yte, pred, k),
        majority=float(np.mean(yte == counts.argmax())),
        n_train=Xtr.shape[0], n_test=Xte.shape[0], n_classes=k,
    )


# ------------------------------------------------------- bootstrap p-values
def bootstrap_pvalues(draws) -> dict:
    """Both p-value conventions from a bootstrap distribution of a difference.

    Reported together and labelled, because the one-sided value is the one that
    answers "is A better than B" and the two-sided value is the one a reviewer
    expects by default.  Quoting a one-sided p for a difference that came out
    the *wrong* way produces numbers like 0.99, which look like a result and
    are not one.

    p_one_sided_greater : P(difference <= 0). Small = A beats B.
    p_two_sided         : 2 * min(P(d <= 0), P(d >= 0)), clipped at 1.
                          Small = A and B differ, in either direction.
    """
    import numpy as _np
    d = _np.asarray(draws, float)
    lo = float(_np.mean(d <= 0))
    hi = float(_np.mean(d >= 0))
    return {"p_one_sided_greater": lo,
            "p_two_sided": float(min(1.0, 2.0 * min(lo, hi)))}
