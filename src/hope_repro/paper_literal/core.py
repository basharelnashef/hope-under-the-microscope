"""Independent implementation of selected printed HOPE v1 equations.

Implements, per the paper's equations:
- effective parameters (Eq 1), surrogate pre-activation law y ~ N(beta, gamma^2) (Eq 78)
- ReLU self-kernel (Eq 3), zero-bias cross-kernel (Eq 5), warped correlation (Eq 4)
- neuron capacity ||f|| = ||w_out|| sqrt(K(i,i)) and layer capacity E = sum ||f||
- pruning and merging costs (Eq 6), distortion rate with static parameter footprint (Eq 23)
- optimal parent: restricted eigen direction (Eq 14) with phase check (Eq 11),
  output direction v* (Eq 13), optimal scale s* (Eq 12)
- physical realization (Eq 15) and BN parameter recovery (Eq 16-18, Appendix D)
- the sequential VGG-8 greedy encoding loop (Sec 10, Algorithms 1-2)

All geometry is numpy; torch is used only for network surgery and evaluation.
The current model surgery covers the adapted sequential VGG-8 diagnostic. It
does not reproduce the paper's ResNet-50/ImageNet experiment.
"""
import numpy as np
import torch

SQ2PI = np.sqrt(2 * np.pi)
BN_EPS = 1e-5
TINY = 1e-30


def _phi(z):
    return np.exp(-0.5 * z * z) / SQ2PI


def _Phi(z):
    from scipy.special import ndtr
    return ndtr(z)


def self_kernel(beta, gamma):
    """Eq 3: K = E[relu(y)^2], y ~ N(beta, gamma^2)."""
    beta = np.asarray(beta, dtype=np.float64)
    g = np.abs(np.asarray(gamma, dtype=np.float64))
    out = np.where(beta > 0, beta, 0.0) ** 2  # gamma -> 0 limit
    ok = g > 1e-12
    c = np.where(ok, beta / np.where(ok, g, 1.0), 0.0)
    val = (g * g + beta * beta) * _Phi(c) + beta * g * _phi(c)
    return np.where(ok, val, out)


def cross_kernel_zero_bias(Kii, Kjj, rho_hat):
    """Eq 5: arc-cosine-1 form scaled by the geometric mean of self-kernels."""
    r = np.clip(rho_hat, -1.0, 1.0)
    return (np.sqrt(np.maximum(1 - r * r, 0.0)) + (np.pi - np.arccos(r)) * r) \
        / np.pi * np.sqrt(np.maximum(Kii * Kjj, 0.0))


def warped_correlation(rho_eff, gamma_i, gamma_j, wnorm_i, wnorm_j):
    """Eq 4: kappa and rho_hat from effective-weight geometry."""
    re = np.clip(rho_eff, -1 + 1e-9, 1 - 1e-9)
    kappa = re / (1 - re * re) \
        * (np.abs(gamma_i) / np.maximum(wnorm_i, TINY)) \
        * (np.abs(gamma_j) / np.maximum(wnorm_j, TINY))
    return 2 * kappa / (1 + np.sqrt(1 + 4 * kappa * kappa))


class UnitLayer:
    """One prunable layer: weights, BN, outgoing slices, and all cached geometry."""

    def __init__(self, name, W_raw, gamma, beta, mu, var, W_out, dP):
        self.name = name
        self.W_raw = W_raw.astype(np.float64)          # (N, n) incoming, flattened
        self.gamma = gamma.astype(np.float64)          # BN affine scale
        self.beta = beta.astype(np.float64)            # BN affine shift
        self.mu = mu.astype(np.float64)                # BN running mean
        self.var = var.astype(np.float64)              # BN running var
        self.W_out = W_out.astype(np.float64)          # (N, c) outgoing, flattened
        self.dP = float(dP)                            # static parameter footprint
        self.N0 = self.W_raw.shape[0]
        self.alive = np.ones(self.N0, dtype=bool)
        self.refresh_all()

    # ---------------- cached geometry ----------------
    def effective(self):
        s = self.gamma / np.sqrt(self.var + BN_EPS)
        W_eff = self.W_raw * s[:, None]
        b_eff = self.beta - s * self.mu
        return W_eff, b_eff

    def refresh_all(self):
        W_eff, b_eff = self.effective()
        self.W_eff, self.b_eff = W_eff, b_eff
        self.Waug = np.concatenate([W_eff, b_eff[:, None]], axis=1)
        self.eff_norm = np.linalg.norm(W_eff, axis=1)
        self.out_norm = np.linalg.norm(self.W_out, axis=1)
        self.Kii = self_kernel(self.beta, self.gamma)
        self.cap = self.out_norm * np.sqrt(np.maximum(self.Kii, 0.0))
        # pair matrices
        self.G_aug = self.Waug @ self.Waug.T
        self.G_out = self.W_out @ self.W_out.T
        G_eff = W_eff @ W_eff.T
        denom = np.outer(np.maximum(self.eff_norm, TINY), np.maximum(self.eff_norm, TINY))
        self.rho_eff = np.clip(G_eff / denom, -1.0, 1.0)
        self.rho_hat = warped_correlation(
            self.rho_eff, self.gamma[:, None], self.gamma[None, :],
            self.eff_norm[:, None], self.eff_norm[None, :])
        self.Kij = cross_kernel_zero_bias(self.Kii[:, None], self.Kii[None, :], self.rho_hat)
        self._pair_geometry(np.arange(self.N0)[:, None], np.arange(self.N0)[None, :], store=True)
        self.E_rem = float(self.cap[self.alive].sum())

    def _pair_geometry(self, I, J, store=False):
        """Vectorized parent direction (Eq 14 + phase check Eq 11) and constants a, b."""
        B11 = self.G_aug[I, I]; B12 = self.G_aug[I, J]; B22 = self.G_aug[J, J]
        O11 = self.G_out[I, I]; O12 = self.G_out[I, J]; O22 = self.G_out[J, J]
        # M = O @ B per pair (2x2); principal right eigenvector
        m11 = O11 * B11 + O12 * B12
        m12 = O11 * B12 + O12 * B22
        m21 = O12 * B11 + O22 * B12
        m22 = O12 * B12 + O22 * B22
        tr = m11 + m22
        det = m11 * m22 - m12 * m21
        lam = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4 * det, 0.0)))
        # eigenvector rows of (M - lam I): pick the better-conditioned expression
        v1a, v2a = m12, lam - m11
        v1b, v2b = lam - m22, m21
        use_a = (np.abs(v1a) + np.abs(v2a)) >= (np.abs(v1b) + np.abs(v2b))
        al = np.where(use_a, v1a, v1b)
        de = np.where(use_a, v2a, v2b)
        # degenerate fallback: pure child i
        bad = (np.abs(al) + np.abs(de)) < 1e-20
        al = np.where(bad, 1.0, al)
        de = np.where(bad, 0.0, de)
        # ambient unit norm: v' B v = 1
        nrm = np.sqrt(np.maximum(al * al * B11 + 2 * al * de * B12 + de * de * B22, TINY))
        al, de = al / nrm, de / nrm
        # phase check: evaluate exact (kernel-level) objective at +-u, keep the better
        objp, bp = self._objective(al, de, I, J)
        objm, bm = self._objective(-al, -de, I, J)
        flip = objm > objp
        al = np.where(flip, -al, al)
        de = np.where(flip, -de, de)
        b = np.where(flip, bm, bp)
        a = self.cap[I] ** 2 + self.cap[J] ** 2
        if store:  # full refresh
            self.alpha, self.delta, self.a_mat, self.b_mat = al, de, a, b
        return al, de, a, b

    def _objective(self, al, de, I, J):
        """Obj(u) = ||K(u,i) w_out_i + K(u,j) w_out_j|| / sqrt(K(u,u)); returns (obj, b)."""
        gi, gj = np.abs(self.gamma[I]), np.abs(self.gamma[J])
        bi, bj = self.beta[I], self.beta[J]
        rh = self.rho_hat[I, J]
        mean_u = al * bi + de * bj
        var_u = np.maximum(al * al * gi * gi + de * de * gj * gj
                           + 2 * al * de * gi * gj * rh, TINY)
        sd_u = np.sqrt(var_u)
        Kuu = np.maximum(self_kernel(mean_u, sd_u), TINY)
        r_ui = np.clip((al * gi * gi + de * gi * gj * rh) / (sd_u * np.maximum(gi, TINY)), -1, 1)
        r_uj = np.clip((de * gj * gj + al * gi * gj * rh) / (sd_u * np.maximum(gj, TINY)), -1, 1)
        Kui = cross_kernel_zero_bias(Kuu, self.Kii[I], r_ui)
        Kuj = cross_kernel_zero_bias(Kuu, self.Kii[J], r_uj)
        z2 = Kui * Kui * self.G_out[I, I] + 2 * Kui * Kuj * self.G_out[I, J] \
            + Kuj * Kuj * self.G_out[J, J]
        b = np.sqrt(np.maximum(z2, 0.0)) / np.sqrt(Kuu)
        return b, b  # objective value IS b (Eq 10)

    # ---------------- costs (Eq 6, Eq 23) ----------------
    def prune_costs(self):
        Nl = int(self.alive.sum())
        Eb = np.maximum(self.E_rem - self.cap, 1e-12)
        J = Nl * self.cap / Eb
        J[~self.alive] = np.inf
        return J

    def merge_costs(self):
        Nl = int(self.alive.sum())
        Erp = self.E_rem - self.cap[:, None] - self.cap[None, :]
        a, b = self.a_mat, self.b_mat
        s = (a + b * np.maximum(Erp, 0.0)) / np.maximum(2 * np.maximum(Erp, 0.0) + b, TINY)
        num = np.sqrt(np.maximum(2 * s * s - 2 * b * s + a, 0.0))
        Eb = np.maximum(Erp + s, 1e-12)
        J = Nl * num / Eb
        dead = ~self.alive
        J[dead, :] = np.inf
        J[:, dead] = np.inf
        np.fill_diagonal(J, np.inf)
        return J, s

    # ---------------- parent generation ----------------
    def make_parent(self, i, j):
        """Compute the printed physical parent for merging ``(i, j)``."""
        a, b = float(self.a_mat[i, j]), float(self.b_mat[i, j])
        Erp = max(self.E_rem - self.cap[i] - self.cap[j], 0.0)
        s = (a + b * Erp) / max(2 * Erp + b, TINY)
        return self._realize_parent_at_scale(i, j, s)

    def _realize_parent_at_scale(self, i, j, scale):
        """Realize Eqs. 15 through 18 at an explicitly selected scale.

        This low-level reconstruction does not select or interpret the scale.
        ``make_parent`` supplies the printed Eq. 12 value. Independent scale
        policies belong outside the paper-literal namespace.
        """
        al, de = float(self.alpha[i, j]), float(self.delta[i, j])
        a, b = float(self.a_mat[i, j]), float(self.b_mat[i, j])
        s = float(scale)
        if not np.isfinite(s) or s < 0:
            raise ValueError("Parent scale must be finite and nonnegative")
        # kernels at u_c
        gi, gj = abs(self.gamma[i]), abs(self.gamma[j])
        rh = float(self.rho_hat[i, j])
        mean_u = al * self.beta[i] + de * self.beta[j]
        var_u = max(al * al * gi * gi + de * de * gj * gj + 2 * al * de * gi * gj * rh, TINY)
        Kuu = max(self_kernel(np.array(mean_u), np.array(np.sqrt(var_u))).item(), TINY)
        r_ui = np.clip((al * gi * gi + de * gi * gj * rh) / (np.sqrt(var_u) * max(gi, TINY)), -1, 1)
        r_uj = np.clip((de * gj * gj + al * gi * gj * rh) / (np.sqrt(var_u) * max(gj, TINY)), -1, 1)
        Kui = cross_kernel_zero_bias(np.array(Kuu), np.array(self.Kii[i]), np.array(r_ui)).item()
        Kuj = cross_kernel_zero_bias(np.array(Kuu), np.array(self.Kii[j]), np.array(r_uj)).item()
        z = Kui * self.W_out[i] + Kuj * self.W_out[j]
        znorm = np.linalg.norm(z)
        if znorm < TINY or s < TINY:
            return None  # degenerate: caller should prune instead
        v_star = z / znorm
        self.last_geometry = dict(al=al, de=de, v_star=v_star, a=a, b=b,
                                  Kuu=Kuu, Kui=Kui, Kuj=Kuj)
        # Eq 15 with the pair's Frobenius ratio
        RF = np.sqrt((self.G_aug[i, i] + self.G_aug[j, j])
                     / max(self.G_out[i, i] + self.G_out[j, j], TINY))
        scale_in = np.sqrt(s * RF) * Kuu ** (-0.25)
        scale_out = np.sqrt(s / RF) * Kuu ** (-0.25)
        u_vec = al * self.Waug[i] + de * self.Waug[j]     # ambient unit norm
        w_aug_p = scale_in * u_vec
        w_eff_p, b_p = w_aug_p[:-1], w_aug_p[-1]
        w_out_p = scale_out * v_star
        # Eq 16-18 recovery
        c1, c2 = scale_in * al, scale_in * de
        beta_p = c1 * self.beta[i] + c2 * self.beta[j]
        g2 = c1 * c1 * self.gamma[i] ** 2 + c2 * c2 * self.gamma[j] ** 2 \
            + 2 * c1 * c2 * gi * gj * rh
        if g2 <= BN_EPS:  # inactive feature (B.5.1)
            return None
        gamma_p = float(np.sqrt(g2))
        mu_p = beta_p - b_p
        var_p = max(g2 - BN_EPS, 0.0)
        w_raw_p = w_eff_p * np.sqrt(var_p + BN_EPS) / gamma_p  # = sign(+1) * w_eff_p
        return dict(w_raw=w_raw_p, gamma=gamma_p, beta=float(beta_p), mu=float(mu_p),
                    var=float(var_p), w_out=w_out_p, s=float(s), a=a, b=b,
                    c1=float(c1), c2=float(c2))

    def apply_merge(self, i, j, parent):
        """Write parent into slot i, kill slot j, update caches (Alg 2 MERGE)."""
        self.E_rem = max(self.E_rem - self.cap[i] - self.cap[j] + parent["s"], 1e-12)
        self.W_raw[i] = parent["w_raw"]
        self.gamma[i] = parent["gamma"]; self.beta[i] = parent["beta"]
        self.mu[i] = parent["mu"]; self.var[i] = parent["var"]
        self.W_out[i] = parent["w_out"]
        self._kill(j)
        self._refresh_unit(i)

    def apply_prune(self, i):
        self.E_rem = max(self.E_rem - self.cap[i], 1e-12)
        self._kill(i)

    def _kill(self, j):
        self.alive[j] = False
        self.W_raw[j] = 0; self.gamma[j] = 0; self.beta[j] = 0
        self.mu[j] = 0; self.var[j] = 1; self.W_out[j] = 0
        self.cap[j] = 0; self.Kii[j] = 0
        for M in (self.G_aug, self.G_out, self.rho_eff, self.rho_hat,
                  self.Kij, self.a_mat, self.b_mat):
            M[j, :] = 0; M[:, j] = 0
        self.Waug[j] = 0; self.W_eff[j] = 0; self.b_eff[j] = 0
        self.eff_norm[j] = 0; self.out_norm[j] = 0

    def _refresh_unit(self, i):
        """Recompute unit i's solitary and pairwise geometry vs all live units."""
        s = self.gamma[i] / np.sqrt(self.var[i] + BN_EPS)
        self.W_eff[i] = self.W_raw[i] * s
        self.b_eff[i] = self.beta[i] - s * self.mu[i]
        self.Waug[i] = np.concatenate([self.W_eff[i], [self.b_eff[i]]])
        self.eff_norm[i] = np.linalg.norm(self.W_eff[i])
        self.out_norm[i] = np.linalg.norm(self.W_out[i])
        self.Kii[i] = self_kernel(np.array(self.beta[i]), np.array(self.gamma[i])).item()
        self.cap[i] = self.out_norm[i] * np.sqrt(max(self.Kii[i], 0.0))
        ga = self.Waug @ self.Waug[i]
        go = self.W_out @ self.W_out[i]
        self.G_aug[i, :] = ga; self.G_aug[:, i] = ga
        self.G_out[i, :] = go; self.G_out[:, i] = go
        ge = self.W_eff @ self.W_eff[i]
        den = np.maximum(self.eff_norm * self.eff_norm[i], TINY)
        re = np.clip(ge / den, -1, 1)
        self.rho_eff[i, :] = re; self.rho_eff[:, i] = re
        rh = warped_correlation(re, self.gamma, self.gamma[i], self.eff_norm, self.eff_norm[i])
        self.rho_hat[i, :] = rh; self.rho_hat[:, i] = rh
        kij = cross_kernel_zero_bias(self.Kii, self.Kii[i], rh)
        self.Kij[i, :] = kij; self.Kij[:, i] = kij
        idx = np.arange(self.N0)
        al, de, a, b = self._pair_geometry(np.full_like(idx, i)[None, :], idx[None, :])
        self.alpha[i, :] = al[0]; self.a_mat[i, :] = a[0]; self.b_mat[i, :] = b[0]
        self.delta[i, :] = de[0]
        # symmetric counterparts: pair (k, i) has roles swapped
        alT, deT, aT, bT = self._pair_geometry(idx[:, None], np.full_like(idx, i)[:, None])
        self.alpha[:, i] = alT[:, 0]; self.delta[:, i] = deT[:, 0]
        self.a_mat[:, i] = aT[:, 0]; self.b_mat[:, i] = bT[:, 0]
        dead = ~self.alive
        for M in (self.a_mat, self.b_mat):
            M[dead, :] = 0; M[:, dead] = 0


# ---------------------------------------------------------------------------
# Model wiring for the Appendix G.2 VGG-8 architecture
# ---------------------------------------------------------------------------

def vgg8_wiring(model):
    """Return list of (name, conv_or_linear, bn, next_module, kind) for prunable layers."""
    f = model.features
    return [
        ("conv0", f[0], f[1], f[4], "conv-conv"),
        ("conv1", f[4], f[5], f[8], "conv-conv"),
        ("conv2", f[8], f[9], f[11], "conv-conv"),
        ("conv3", f[11], f[12], f[15], "conv-conv"),
        ("conv4", f[15], f[16], f[18], "conv-conv"),
        ("conv5", f[18], f[19], model.bottleneck, "conv-dense"),
        ("dense", model.bottleneck, model.bn_b, model.head, "dense-dense"),
    ]


def extract_layers(model):
    layers = []
    for name, op, bn, nxt, kind in vgg8_wiring(model):
        W = op.weight.detach().cpu().numpy()
        N = W.shape[0]
        W_raw = W.reshape(N, -1)
        if kind == "conv-conv":
            nW = nxt.weight.detach().cpu().numpy()      # (Cout, N, k, k)
            W_out = nW[:, :N].transpose(1, 0, 2, 3).reshape(N, -1)
        else:                                            # next is Linear (in_features = N)
            nW = nxt.weight.detach().cpu().numpy()      # (out, N)
            W_out = nW.T.copy()
        dP = W_raw.shape[1] + W_out.shape[1] + 4        # B.2.1 footprint
        layers.append(UnitLayer(
            name, W_raw,
            bn.weight.detach().cpu().numpy(), bn.bias.detach().cpu().numpy(),
            bn.running_mean.detach().cpu().numpy(), bn.running_var.detach().cpu().numpy(),
            W_out, dP))
    return layers


def write_back_unit(model, li, i, L):
    """Push numpy state of unit (li, i) into the torch model."""
    wiring = vgg8_wiring(model)
    name, op, bn, nxt, kind = wiring[li]
    with torch.no_grad():
        t = lambda x: torch.tensor(x, dtype=op.weight.dtype, device=op.weight.device)
        op.weight[i] = t(L.W_raw[i].reshape(op.weight[i].shape))
        bn.weight[i] = t(L.gamma[i]); bn.bias[i] = t(L.beta[i])
        bn.running_mean[i] = t(L.mu[i]); bn.running_var[i] = t(max(L.var[i], 0.0))
        if kind == "conv-conv":
            N = L.N0
            nxt.weight[:, i] = t(L.W_out[i].reshape(nxt.weight.shape[0],
                                                    *nxt.weight.shape[2:]))
        else:
            nxt.weight[:, i] = t(L.W_out[i])


class HopeEncoder:
    """Greedy progressive encoder (Sec 10 / Algorithm 1)."""

    def __init__(self, model, use_merge=True, *, min_alive=2, use_prune=True):
        self.model = model
        self.layers = extract_layers(model)
        self.use_merge = use_merge
        self.use_prune = use_prune
        self.min_alive = min_alive
        self.log = []

    def _make_parent(self, layer, first, second):
        """Select the printed parent construction for one merge."""
        return layer.make_parent(first, second)

    @property
    def total_alive(self):
        return sum(int(L.alive.sum()) for L in self.layers)

    @property
    def total_units(self):
        return sum(L.N0 for L in self.layers)

    def step(self):
        """One greedy action; returns False when nothing admissible remains."""
        best = None
        for li, L in enumerate(self.layers):
            if int(L.alive.sum()) <= self.min_alive:
                continue
            if self.use_prune:
                Jp = L.prune_costs()
                i = int(np.argmin(Jp))
                if np.isfinite(Jp[i]):
                    dr = Jp[i] / L.dP
                    if best is None or dr < best[0]:
                        best = (dr, "prune", li, i, -1, float(Jp[i]))
            if self.use_merge:
                Jm, _ = L.merge_costs()
                k = int(np.argmin(Jm))
                mi, mj = divmod(k, L.N0)
                if np.isfinite(Jm[mi, mj]):
                    dr = Jm[mi, mj] / L.dP
                    if best is None or dr < best[0]:
                        best = (dr, "merge", li, mi, mj, float(Jm[mi, mj]))
        if best is None:
            return False
        dr, kind, li, i, j, J = best
        L = self.layers[li]
        if kind == "merge":
            parent = self._make_parent(L, i, j)
            if parent is None:  # degenerate merge: prune the weaker child instead
                w = i if L.cap[i] <= L.cap[j] else j
                L.apply_prune(w)
                write_back_unit(self.model, li, w, L)
                self.log.append(("prune*", li, w, J, dr))
                return True
            L.apply_merge(i, j, parent)
            write_back_unit(self.model, li, i, L)
            write_back_unit(self.model, li, j, L)
            self.log.append(("merge", li, (i, j), J, dr))
        else:
            L.apply_prune(i)
            write_back_unit(self.model, li, i, L)
            self.log.append(("prune", li, i, J, dr))
        return True


# ---------------------------------------------------------------------------
# Magnitude baselines (Sec 11.1)
# ---------------------------------------------------------------------------

def baseline_order(model, kind):
    """Static global ascending ranking. kind in {l1_in, l1_joint, bn_scale}."""
    layers = extract_layers(model)
    scores = []
    for li, L in enumerate(layers):
        if kind == "l1_in":
            s = np.abs(L.W_raw).sum(axis=1)
        elif kind == "l1_joint":
            s = np.abs(L.W_raw).sum(axis=1) + np.abs(L.W_out).sum(axis=1)
        elif kind == "bn_scale":
            s = np.abs(L.gamma)
        else:
            raise ValueError(kind)
        for i in range(L.N0):
            scores.append((s[i], li, i))
    scores.sort()
    return layers, scores


def prune_unit_torch(model, li, i):
    """Zero a unit in the torch model directly (baseline execution)."""
    name, op, bn, nxt, kind = vgg8_wiring(model)[li]
    with torch.no_grad():
        op.weight[i] = 0
        bn.weight[i] = 0; bn.bias[i] = 0
        bn.running_mean[i] = 0; bn.running_var[i] = 1
        nxt.weight[:, i] = 0


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    c = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        c += (model(x).argmax(1) == y).sum().item()
        n += y.numel()
    return c / n


# ---------------------------------------------------------------------------
# Self-test: verify the surgery algebra inside a real torch model
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(1); np.random.seed(1)
    from .vgg8 import VGG8
    model = VGG8().eval()
    # give BN nontrivial running stats and affine params
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.running_mean.uniform_(-0.4, 0.4)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.4, 1.4)
                m.bias.uniform_(-0.5, 0.5)

    # Test 1: effective-parameter identity on the dense layer
    L = extract_layers(model)[6]
    x = np.random.randn(512)
    y_np = L.W_eff @ x + L.b_eff
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)[None]
        y_t = model.bn_b(model.bottleneck(xt))[0].numpy()
    e1 = np.abs(y_np - y_t).max()
    print(f"[test1] effective-params identity max err: {e1:.2e}")
    assert e1 < 1e-4

    # Test 2: exact-twin merge in torch - paper scale halves the contribution
    li, i, j = 6, 3, 7
    wiring = vgg8_wiring(model)
    name, op, bn, nxt, kind = wiring[li]
    with torch.no_grad():  # make j an exact twin of i
        op.weight[j] = op.weight[i]
        bn.weight[j] = bn.weight[i]; bn.bias[j] = bn.bias[i]
        bn.running_mean[j] = bn.running_mean[i]; bn.running_var[j] = bn.running_var[i]
        nxt.weight[:, j] = nxt.weight[:, i]

    def head_contrib(m, X):
        with torch.no_grad():
            z = m.bn_b(m.bottleneck(X)).relu()
            return z @ m.head.weight.T  # (B, 10) pre-bias

    X = torch.randn(4096, 512)
    ref = head_contrib(model, X)
    import copy
    m2 = copy.deepcopy(model)
    enc = HopeEncoder(m2, use_merge=True)
    L2 = enc.layers[li]
    par = L2.make_parent(i, j)
    L2.apply_merge(i, j, par)
    write_back_unit(m2, li, i, L2)
    write_back_unit(m2, li, j, L2)
    out = head_contrib(m2, X)
    derr = (out - ref).norm() / ref.norm()
    print(f"[test2] twin merge (paper s*): s={par['s']:.4f} "
          f"b={par['b']:.4f} relative network change {derr:.4f}")
    # expected: paper s* is approximately b/2 and changes the network

    # Test 3: forward consistency of recovered BN params after a random-pair merge
    m3 = copy.deepcopy(model)
    enc3 = HopeEncoder(m3, use_merge=True)
    L3 = enc3.layers[6]
    pi, pj = 11, 42
    par = L3.make_parent(pi, pj)
    L3.apply_merge(pi, pj, par)
    write_back_unit(m3, 6, pi, L3)
    write_back_unit(m3, 6, pj, L3)
    x = np.random.randn(512)
    y_pred = L3.W_eff[pi] @ x + L3.b_eff[pi]
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)[None]
        y_t = m3.bn_b(m3.bottleneck(xt))[0, pi].item()
    e3 = abs(y_pred - y_t)
    print(f"[test3] BN recovery forward identity err: {e3:.2e} (s*={par['s']:.4f})")
    assert e3 < 1e-3
    print("self-test passed")


if __name__ == "__main__":
    _self_test()
