"""A tiny CPU-trainable linear-softmax policy shared by BC and DPO.

The design constraint is the project's spine: the class gives API credits, not
GPUs, so the *policy* must train on a laptop in seconds. A multinomial logistic
model over the nine NodeFeatures is enough -- the expensive signal is already
baked into the traces. Keeping BC and DPO on the SAME scorer is deliberate: it
makes the comparison clean (same capacity, same features), so any difference is
attributable to the learning objective, not the model.

  - BC fits cross-entropy on (features -> chosen action), weighted by the
    realized value-per-cost so good decisions dominate.
  - DPO fits a pairwise log-sigmoid preference loss against a frozen reference
    policy (the BC model), exactly the DPO objective, in the contextual-bandit
    special case where each datapoint is a single decision.

Both are plain numpy gradient descent. No sklearn, no torch, no GPU.
"""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LinearSoftmaxPolicy:
    """Logits = standardized_features @ W.T + b, over a fixed ordered action set."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        l2: float = 1e-3,
        lr: float = 0.2,
        epochs: int = 400,
        seed: int | None = None,
    ) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.01, size=(n_actions, n_features))
        self.b = np.zeros(n_actions)
        # Feature standardization, fit on the training matrix.
        self.mu = np.zeros(n_features)
        self.sigma = np.ones(n_features)
        self._fitted = False

    # ---- feature scaling -------------------------------------------------
    def set_scaler(self, X: np.ndarray) -> None:
        self.mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma[sigma < 1e-6] = 1.0
        self.sigma = sigma

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    # ---- inference -------------------------------------------------------
    def logits(self, X: np.ndarray) -> np.ndarray:
        return self._scale(np.atleast_2d(X)) @ self.W.T + self.b

    def probs(self, x: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(x))[0]

    # ---- behavior cloning ------------------------------------------------
    def fit_bc(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        """Weighted multinomial cross-entropy via full-batch gradient descent."""
        X = np.atleast_2d(X).astype(np.float64)
        y = np.asarray(y, dtype=int)
        n = len(X)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        w = w / (w.sum() + 1e-12) * n  # normalize so lr is scale-stable

        self.set_scaler(X)
        Xs = self._scale(X)
        onehot = np.zeros((n, self.n_actions))
        onehot[np.arange(n), y] = 1.0

        for _ in range(self.epochs):
            p = _softmax(Xs @ self.W.T + self.b)
            g = (p - onehot) * w[:, None]          # (n, A)
            dW = g.T @ Xs / n + self.l2 * self.W
            db = g.mean(axis=0)
            self.W -= self.lr * dW
            self.b -= self.lr * db
        self._fitted = True

    # ---- KTO -------------------------------------------------------------
    def fit_kto(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        desirable: np.ndarray,
        reference: "LinearSoftmaxPolicy",
        *,
        beta: float = 0.1,
        lambda_d: float = 1.0,
        lambda_u: float = 1.0,
    ) -> None:
        """Kahneman-Tversky Optimization in the contextual-bandit reduction.

        Unlike DPO, KTO needs no preference *pairs*: each logged decision is an
        independent (state, action) example tagged desirable or undesirable. That
        is exactly what our traces give us (a decision's realized value-per-cost
        thresholded), so KTO sidesteps the bucket-and-pair approximation DPO leans
        on -- and uses every decision, which matters in our tiny-trace regime.

        For example (x, a) with implicit reward
            r = beta * (logp_theta(a|x) - logp_ref(a|x))
        and a batch KL baseline z = clamp(mean_batch(r_detached), min=0):
            desirable:   L = lambda_d * (1 - sigmoid(beta*(r - z)))
            undesirable: L = lambda_u * (1 - sigmoid(beta*(z - r)))
        Gradients flow through logp_theta(a|x) only (z is detached, recomputed per
        epoch). We warm-start from the reference and share its scaler, like DPO, so
        the implicit-reward anchoring is meaningful and the BC/DPO/KTO comparison
        stays on equal footing (same model, same features, same reference)."""
        X = np.atleast_2d(X).astype(np.float64)
        actions = np.asarray(actions, dtype=int)
        desirable = np.asarray(desirable, dtype=bool)
        n = len(X)

        self.mu, self.sigma = reference.mu.copy(), reference.sigma.copy()
        self.W, self.b = reference.W.copy(), reference.b.copy()
        Xs = self._scale(X)

        ref_logp = _log_softmax(Xs @ reference.W.T + reference.b)
        ref_logp_a = ref_logp[np.arange(n), actions]
        idx = np.arange(n)

        for _ in range(self.epochs):
            logits = Xs @ self.W.T + self.b
            p = _softmax(logits)
            logp = _log_softmax(logits)
            r = beta * (logp[idx, actions] - ref_logp_a)        # implicit reward
            z = max(0.0, float(r.mean()))                       # detached KL baseline

            # dL/dr per example (z held constant this step).
            u = beta * (r - z)
            v = beta * (z - r)
            dL_dr = np.where(
                desirable,
                -lambda_d * beta * _sigmoid(u) * _sigmoid(-u),
                +lambda_u * beta * _sigmoid(v) * _sigmoid(-v),
            )
            # dr/dlogit_k = beta * ([k==a] - p_k); chain through dL/dr.
            g = -p * (dL_dr * beta)[:, None]                    # (n, A), the -p_k term
            g[idx, actions] += (dL_dr * beta)                   # add the [k==a] term
            dW = g.T @ Xs / n + self.l2 * self.W
            db = g.mean(axis=0)
            self.W -= self.lr * dW
            self.b -= self.lr * db
        self._fitted = True

    # ---- DPO -------------------------------------------------------------
    def fit_dpo(
        self,
        X: np.ndarray,
        a_pref: np.ndarray,
        a_rej: np.ndarray,
        reference: "LinearSoftmaxPolicy",
        *,
        beta: float = 0.1,
    ) -> None:
        """Pairwise DPO loss against a frozen reference, anchored at each pair's
        state. For pair (x, a_w, a_l):

            margin(theta) = logit_theta(x, a_w) - logit_theta(x, a_l)
            h = beta * (margin(theta) - margin(ref))
            loss = -log sigmoid(h)

        We initialize from the reference and share its scaler so the KL-style
        anchoring is meaningful. This is the contextual-bandit reduction of DPO:
        each logged decision is one (state, action) datum, and preferred/rejected
        actions are mined from realized value-per-cost (see DPOAllocator)."""
        X = np.atleast_2d(X).astype(np.float64)
        a_pref = np.asarray(a_pref, dtype=int)
        a_rej = np.asarray(a_rej, dtype=int)
        n = len(X)

        # Inherit the reference's fitted scaler and warm-start from it.
        self.mu, self.sigma = reference.mu.copy(), reference.sigma.copy()
        self.W, self.b = reference.W.copy(), reference.b.copy()
        Xs = self._scale(X)

        # Reference margins are constant during training.
        ref_logits = Xs @ reference.W.T + reference.b
        ref_margin = ref_logits[np.arange(n), a_pref] - ref_logits[np.arange(n), a_rej]

        for _ in range(self.epochs):
            logits = Xs @ self.W.T + self.b
            margin = logits[np.arange(n), a_pref] - logits[np.arange(n), a_rej]
            h = beta * (margin - ref_margin)
            # dloss/dmargin = -beta * sigmoid(-h)
            coeff = -beta * _sigmoid(-h)            # (n,)
            dW = np.zeros_like(self.W)
            db = np.zeros_like(self.b)
            for k in range(self.n_actions):
                sel_w = (a_pref == k).astype(np.float64)
                sel_l = (a_rej == k).astype(np.float64)
                s = coeff * (sel_w - sel_l)         # +x for preferred, -x for rejected
                dW[k] = s @ Xs / n + self.l2 * self.W[k]
                db[k] = s.mean()
            self.W -= self.lr * dW
            self.b -= self.lr * db
        self._fitted = True
