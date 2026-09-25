"""Phase 3: representation geometry.

Concept Activation Vectors, subspace overlap between face regions (the binding
question), intrinsic dimensionality of the state code, and single-narrative
trajectories through a reduced space.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def concept_direction(X: np.ndarray, y: np.ndarray, a: int, b: int,
                      normalize: bool = True) -> np.ndarray:
    """CAV as the difference of class means: mean(a) - mean(b)."""
    va = X[y == a].mean(axis=0)
    vb = X[y == b].mean(axis=0)
    d = va - vb
    if normalize:
        n = np.linalg.norm(d)
        if n > 0:
            d = d / n
    return d


def probe_direction(X: np.ndarray, y: np.ndarray, cls: int) -> np.ndarray:
    """One-vs-rest logistic-regression weight vector for `cls`.

    A supervised alternative to the difference-of-means CAV; it discounts
    directions of high nuisance variance, so the two agreeing is evidence the
    direction is real rather than an artifact of class-conditional means.
    """
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(sc.transform(X), (y == cls).astype(int))
    w = clf.coef_[0] / sc.scale_
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def direction_bank(X: np.ndarray, labels: dict[str, np.ndarray],
                   classes: dict[str, list[str]],
                   method: str = "cav") -> dict[str, np.ndarray]:
    """One direction per (target, class), e.g. 'lips.finish=dewy'."""
    out: dict[str, np.ndarray] = {}
    for target, y in labels.items():
        for ci, cname in enumerate(classes[target]):
            d = (_one_vs_rest_cav(X, y, ci) if method == "cav"
                 else probe_direction(X, y, ci))
            out[f"{target}={cname}"] = d
    return out


def _one_vs_rest_cav(X: np.ndarray, y: np.ndarray, cls: int) -> np.ndarray:
    va = X[y == cls].mean(axis=0)
    vb = X[y != cls].mean(axis=0)
    d = va - vb
    n = np.linalg.norm(d)
    return d / n if n > 0 else d


def similarity_matrix(bank: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    keys = list(bank)
    M = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            M[i, j] = cosine(bank[a], bank[b])
    return keys, M


def subspace_principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between the column spaces of A and B.

    Used for the binding question: if the lip-colour subspace and the
    eye-colour subspace are near-orthogonal, the model keeps regions separate;
    small angles mean the attribute is represented region-agnostically.
    """
    qa, _ = np.linalg.qr(A)
    qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def target_subspace(X: np.ndarray, y: np.ndarray, classes: list[str],
                    method: str = "cav") -> np.ndarray:
    """(hidden, n_classes) matrix whose columns span the target's code."""
    cols = []
    for ci in range(len(classes)):
        if (y == ci).sum() < 2 or (y != ci).sum() < 2:
            continue
        cols.append(_one_vs_rest_cav(X, y, ci) if method == "cav"
                    else probe_direction(X, y, ci))
    return np.stack(cols, axis=1)


def dimensionality_curve(X: np.ndarray, y: np.ndarray,
                         n_components: tuple[int, ...] = (1, 2, 3, 5, 8, 16, 32,
                                                          64, 128, 256),
                         seed: int = 0) -> list[dict]:
    """Linear-probe accuracy as a function of PCA dimensionality.

    Low-dimensional separation is a stronger claim than needing the full
    residual stream: it says the state occupies a compact subspace.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    Xs_tr, Xs_te = sc.transform(X[tr]), sc.transform(X[te])
    pca = PCA(n_components=min(max(n_components), Xs_tr.shape[1],
                               len(tr) - 1), random_state=seed).fit(Xs_tr)
    Ztr, Zte = pca.transform(Xs_tr), pca.transform(Xs_te)
    out = []
    for k in n_components:
        if k > Ztr.shape[1]:
            continue
        clf = LogisticRegression(max_iter=500, n_jobs=-1)
        clf.fit(Ztr[:, :k], y[tr])
        out.append({
            "n_components": k,
            "accuracy": float(np.mean(clf.predict(Zte[:, :k]) == y[te])),
            "explained_variance": float(
                pca.explained_variance_ratio_[:k].sum()),
        })
    return out


def fit_projection(X: np.ndarray, n_components: int = 3, seed: int = 0):
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=n_components, random_state=seed).fit(sc.transform(X))

    def project(Y: np.ndarray) -> np.ndarray:
        return pca.transform(sc.transform(Y))

    return project, pca
