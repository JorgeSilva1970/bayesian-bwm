"""
Motor estatístico do Bayesian Best-Worst Method (BWM).

Referências principais
----------------------
- Rezaei, J. (2015). Best-worst multi-criteria decision-making method. Omega, 53, 49-57.
- Rezaei, J. (2016). Best-worst MCDM method: Some properties and a linear model. Omega, 64, 126-130.
- Mohammadi, M. & Rezaei, J. (2020). Bayesian best-worst method: A probabilistic group
  decision making model. Omega, 96, 102075.
- Liang, F., Brunelli, M. & Rezaei, J. (2020). Consistency issues in the best worst method:
  Measurements and thresholds. Omega, 96, 102175.

Modelo hierárquico (Mohammadi & Rezaei, 2020)
---------------------------------------------
Para cada decisor k = 1..K:
    A_B^k | w^k ~ Multinomial( (1/w^k) / sum(1/w^k) )   # vetor Best-to-Others
    A_W^k | w^k ~ Multinomial( w^k )                     # vetor Others-to-Worst
    w^k   | w*, γ ~ Dirichlet( γ · w* )                  # pesos individuais
    γ              ~ Gamma(a=0.01, b=0.01)               # concentração (quanto os decisores concordam)
    w*             ~ Dirichlet( 1 )                      # pesos agregados do grupo

A amostragem é feita por Hamiltonian Monte Carlo (HMC) com gradientes analíticos, adaptação
do passo (dual averaging) e matriz de massa diagonal, no espaço não restrito (transformação
log-rácio aditiva para os simplexos, log para γ). Não depende de PyMC/Stan, o que mantém a app
leve para o Render.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.special import digamma, gammaln

# ---------------------------------------------------------------------------
# Transformações simplex <-> espaço não restrito
# ---------------------------------------------------------------------------


def _to_simplex(z: np.ndarray) -> np.ndarray:
    """Converte z (..., n-1) em w (..., n) no simplex (softmax com última componente = 0)."""
    zfull = np.concatenate([z, np.zeros(z.shape[:-1] + (1,))], axis=-1)
    zfull = zfull - zfull.max(axis=-1, keepdims=True)
    e = np.exp(zfull)
    return e / e.sum(axis=-1, keepdims=True)


def _from_simplex(w: np.ndarray) -> np.ndarray:
    """Inversa: w (..., n) -> z (..., n-1)."""
    w = np.clip(w, 1e-300, None)
    return np.log(w[..., :-1]) - np.log(w[..., -1:])


# ---------------------------------------------------------------------------
# Densidades (log) usadas pelo amostrador
# ---------------------------------------------------------------------------


def _loglik(w: np.ndarray, AB: np.ndarray, AW: np.ndarray) -> np.ndarray:
    """Log-verosimilhança multinomial (sem constante) para cada decisor. w: (K, n)."""
    logw = np.log(w)
    inv = 1.0 / w
    log_pb = -logw - np.log(inv.sum(axis=-1, keepdims=True))
    return (AB * log_pb).sum(axis=-1) + (AW * logw).sum(axis=-1)


def _logdir(w: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Log-densidade Dirichlet(alpha) avaliada em w (vetorizada na última dimensão)."""
    return (
        gammaln(alpha.sum(axis=-1))
        - gammaln(alpha).sum(axis=-1)
        + ((alpha - 1.0) * np.log(w)).sum(axis=-1)
    )


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------


@dataclass
class BayesBWMResult:
    criteria: list[str]
    decision_makers: list[str]
    w_star: np.ndarray            # (chains*draws, n) amostras dos pesos agregados
    w_ind: np.ndarray             # (chains*draws, K, n) amostras dos pesos individuais
    gamma: np.ndarray             # (chains*draws,)
    chains: int
    draws: int
    acceptance: dict = field(default_factory=dict)

    # ---- resumos -----------------------------------------------------------
    def summary(self, cred: float = 0.95) -> "dict[str, np.ndarray]":
        lo, hi = (1 - cred) / 2 * 100, (1 + cred) / 2 * 100
        return {
            "media": self.w_star.mean(axis=0),
            "desvio": self.w_star.std(axis=0, ddof=1),
            "mediana": np.median(self.w_star, axis=0),
            "ic_inf": np.percentile(self.w_star, lo, axis=0),
            "ic_sup": np.percentile(self.w_star, hi, axis=0),
        }

    def credal_matrix(self) -> np.ndarray:
        """P[i, j] = probabilidade de o critério i ser mais importante que o critério j."""
        W = self.w_star
        n = W.shape[1]
        P = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                if i != j:
                    P[i, j] = float(np.mean(W[:, i] > W[:, j]))
        return P

    def rank_probabilities(self) -> np.ndarray:
        """R[i, r] = probabilidade de o critério i ocupar a posição r (0 = mais importante)."""
        order = np.argsort(-self.w_star, axis=1)
        n = self.w_star.shape[1]
        R = np.zeros((n, n))
        for r in range(n):
            counts = np.bincount(order[:, r], minlength=n)
            R[:, r] = counts / order.shape[0]
        return R

    def individual_means(self) -> np.ndarray:
        return self.w_ind.mean(axis=0)  # (K, n)

    def diagnostics(self) -> "dict[str, np.ndarray]":
        W = self.w_star.reshape(self.chains, self.draws, -1)
        return {"rhat": split_rhat(W), "ess": ess(W)}


# ---------------------------------------------------------------------------
# Amostrador MCMC — Hamiltonian Monte Carlo com gradientes analíticos
# ---------------------------------------------------------------------------


class _Posterior:
    """Log-posterior conjunto e respetivo gradiente no espaço não restrito."""

    def __init__(self, AB, AW, a_g, b_g):
        self.AB, self.AW = AB, AW
        self.K, self.n = AB.shape
        self.NB = AB.sum(axis=1)
        self.a_g, self.b_g = a_g, b_g
        self.d_ind = self.K * (self.n - 1)
        self.dim = self.d_ind + (self.n - 1) + 1

    def unpack(self, theta):
        K, n = self.K, self.n
        z_ind = theta[: self.d_ind].reshape(K, n - 1)
        z_star = theta[self.d_ind: self.d_ind + n - 1]
        u = theta[-1]
        return _to_simplex(z_ind), _to_simplex(z_star), u

    def pack(self, w_ind, w_star, u):
        return np.concatenate([_from_simplex(w_ind).ravel(), _from_simplex(w_star), [u]])

    def logp_grad(self, theta):
        with np.errstate(all="ignore"):
            return self._logp_grad(theta)

    def _logp_grad(self, theta):
        AB, AW, K, n = self.AB, self.AW, self.K, self.n
        w, ws, u = self.unpack(theta)
        gam = np.exp(u)
        logw, logws = np.log(w), np.log(ws)
        inv = 1.0 / w
        S = inv.sum(axis=1)
        alpha = gam * ws

        lp = (
            (-(AB * logw)).sum() - (self.NB * np.log(S)).sum() + (AW * logw).sum()
            + K * (gammaln(gam) - gammaln(alpha).sum()) + ((alpha - 1) * logw).sum()
            + logw.sum()                     # jacobiano de w^k
            + logws.sum()                    # jacobiano de w* (prior Dir(1) é constante)
            + (self.a_g - 1) * u - self.b_g * gam + u
        )
        if not np.isfinite(lp) or np.any(w <= 0) or np.any(ws <= 0):
            return -np.inf, np.zeros_like(theta)

        # gradiente em relação a w (antes da cadeia softmax)
        g_w = (-AB + AW + alpha) * inv + (self.NB / S)[:, None] * inv ** 2
        psi_a = digamma(alpha)
        g_ws = gam * (-K * psi_a + logw.sum(axis=0)) + 1.0 / ws
        dgam = K * digamma(gam) - K * (ws * psi_a).sum() + (ws * logw).sum()
        g_u = gam * dgam + (self.a_g - 1) - self.b_g * gam + 1

        # cadeia softmax: dL/dx_i = w_i (g_i − Σ w_j g_j)
        gz_ind = (w * (g_w - (w * g_w).sum(axis=1, keepdims=True)))[:, :-1]
        gz_star = (ws * (g_ws - (ws * g_ws).sum())) [:-1]
        grad = np.concatenate([gz_ind.ravel(), gz_star, [g_u]])
        if not np.all(np.isfinite(grad)):
            return -np.inf, np.zeros_like(theta)
        return lp, grad


def _hmc_chain(post, theta0, draws, burn_in, rng, target_accept=0.8, max_steps=64,
               progress=None):
    dim = post.dim
    theta = theta0.copy()
    lp, g = post.logp_grad(theta)
    inv_mass = np.ones(dim)
    eps = 0.05
    # dual averaging (Hoffman & Gelman, 2014)
    mu, h_bar, log_eps_bar, t0, gamma_da, kappa = np.log(10 * eps), 0.0, 0.0, 10, 0.05, 0.75
    m = 0  # contador do dual averaging (reiniciado a cada janela de adaptação)
    traj_len = 1.0
    window_samples = []
    windows = _adapt_windows(burn_in)
    out = np.empty((draws, dim))
    acc_sum, div = 0.0, 0
    total = burn_in + draws

    for it in range(total):
        p0 = rng.normal(size=dim) / np.sqrt(inv_mass)
        n_steps = int(np.clip(np.ceil(traj_len / eps * rng.uniform(0.6, 1.4)), 1, max_steps))
        th, p, gg = theta.copy(), p0.copy(), g.copy()
        p += 0.5 * eps * gg
        ok = True
        for s in range(n_steps):
            th += eps * inv_mass * p
            lp_new, gg = post.logp_grad(th)
            if not np.isfinite(lp_new):
                ok = False
                break
            if s < n_steps - 1:
                p += eps * gg
        if ok:
            p += 0.5 * eps * gg
            h0 = -lp + 0.5 * np.sum(inv_mass * p0 ** 2)
            h1 = -lp_new + 0.5 * np.sum(inv_mass * p ** 2)
            log_a = min(0.0, h0 - h1) if np.isfinite(h1) else -np.inf
            if h1 - h0 > 1000 and it >= burn_in:
                div += 1
        else:
            log_a = -np.inf
            div += it >= burn_in
        a = float(np.exp(log_a))
        if np.log(rng.uniform()) < log_a:
            theta, lp, g = th, lp_new, gg

        if it < burn_in:
            m += 1
            h_bar = (1 - 1 / (m + t0)) * h_bar + (target_accept - a) / (m + t0)
            log_eps = mu - np.sqrt(m) / gamma_da * h_bar
            eta = m ** (-kappa)
            log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
            eps = float(np.exp(log_eps))
            # adaptação da matriz de massa diagonal por janelas
            for (ws_, we_) in windows:
                if ws_ <= it < we_:
                    window_samples.append(theta.copy())
                if it == we_ - 1 and len(window_samples) > 10:
                    arr = np.array(window_samples)
                    nw = arr.shape[0]
                    var = arr.var(axis=0, ddof=1)
                    inv_mass = (nw / (nw + 5)) * var + 1e-3 * (5 / (nw + 5))
                    window_samples = []
                    mu = np.log(10 * eps)
                    h_bar, log_eps_bar, m = 0.0, 0.0, 0
            if it == burn_in - 1:
                eps = float(np.exp(log_eps_bar))
        else:
            out[it - burn_in] = theta
            acc_sum += a
        if progress and it % 100 == 0:
            progress(it / total)
    return out, acc_sum / max(draws, 1), div, eps


def _adapt_windows(burn_in):
    """Janelas de adaptação da massa (esquema semelhante ao Stan)."""
    init, term = int(0.15 * burn_in), int(0.1 * burn_in)
    start, end = init, burn_in - term
    windows, size = [], 25
    while start < end:
        stop = min(start + size, end)
        if end - stop < 2 * size:
            stop = end
        windows.append((start, stop))
        start, size = stop, size * 2
    return windows


def bayesian_bwm(
    AB: np.ndarray,
    AW: np.ndarray,
    criteria: list[str] | None = None,
    decision_makers: list[str] | None = None,
    draws: int = 3000,
    burn_in: int = 1500,
    chains: int = 3,
    gamma_prior: tuple[float, float] = (0.01, 0.01),
    seed: int | None = 42,
    progress_callback=None,
) -> BayesBWMResult:
    """
    Estima o Bayesian BWM por HMC.

    AB : (K, n) vetores Best-to-Others (1..9), um por decisor.
    AW : (K, n) vetores Others-to-Worst (1..9), um por decisor.
    """
    AB = np.atleast_2d(np.asarray(AB, dtype=float))
    AW = np.atleast_2d(np.asarray(AW, dtype=float))
    if AB.shape != AW.shape:
        raise ValueError("As matrizes Best-to-Others e Others-to-Worst têm dimensões diferentes.")
    K, n = AB.shape
    if n < 2:
        raise ValueError("São necessários pelo menos 2 critérios.")
    criteria = criteria or [f"C{j + 1}" for j in range(n)]
    decision_makers = decision_makers or [f"Decisor {k + 1}" for k in range(K)]

    post = _Posterior(AB, AW, *gamma_prior)
    rng = np.random.default_rng(seed)
    W_star, W_ind, G = [], [], []
    acc, divs = [], 0

    for c in range(chains):
        w0 = np.array([_approx_weights(AB[k], AW[k]) for k in range(K)])
        w0 = 0.8 * w0 + 0.2 * rng.dirichlet(np.ones(n), size=K)
        ws0 = w0.mean(axis=0)
        theta0 = post.pack(w0, ws0, np.log(rng.uniform(5, 30)))

        def prog(f, c=c):
            if progress_callback:
                progress_callback((c + f) / chains)

        samples, a, dv, _ = _hmc_chain(post, theta0, draws, burn_in, rng, progress=prog)
        acc.append(a)
        divs += dv
        for th in samples:
            wi, wsr, u = post.unpack(th)
            W_ind.append(wi)
            W_star.append(wsr)
            G.append(np.exp(u))

    if progress_callback:
        progress_callback(1.0)

    return BayesBWMResult(
        criteria=list(criteria),
        decision_makers=list(decision_makers),
        w_star=np.array(W_star),
        w_ind=np.array(W_ind),
        gamma=np.array(G),
        chains=chains,
        draws=draws,
        acceptance={"taxa_aceitacao": float(np.mean(acc)), "divergencias": int(divs)},
    )


def _approx_weights(ab: np.ndarray, aw: np.ndarray) -> np.ndarray:
    """Ponto de partida simples: média geométrica de 1/a_Bj e a_jW, normalizada."""
    w = np.sqrt((1.0 / ab) * aw)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Diagnósticos de convergência
# ---------------------------------------------------------------------------


def split_rhat(x: np.ndarray) -> np.ndarray:
    """R-hat com divisão de cadeias (Gelman et al.). x: (chains, draws, p)."""
    c, d, p = x.shape
    half = d // 2
    if half < 2:
        return np.full(p, np.nan)
    x = np.concatenate([x[:, :half], x[:, half: 2 * half]], axis=0)  # (2c, half, p)
    m, nn = x.shape[0], x.shape[1]
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    B = nn * chain_means.var(axis=0, ddof=1)
    Wv = chain_vars.mean(axis=0)
    var_hat = (nn - 1) / nn * Wv + B / nn
    return np.sqrt(var_hat / Wv)


def ess(x: np.ndarray) -> np.ndarray:
    """Tamanho efetivo da amostra (estimador de Geyer, sequência inicial positiva)."""
    c, d, p = x.shape
    out = np.empty(p)
    for j in range(p):
        y = x[:, :, j]
        y = y - y.mean(axis=1, keepdims=True)
        var = y.var(axis=1).mean()
        if var <= 0:
            out[j] = np.nan
            continue
        nfft = 1 << (2 * d - 1).bit_length()
        f = np.fft.rfft(y, n=nfft, axis=1)
        acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=1)[:, :d] / d
        rho = acov.mean(axis=0) / var
        s, t = 0.0, 0
        while t + 1 < d:
            pair = rho[t] + rho[t + 1]
            if pair < 0:
                break
            s += pair
            t += 2
        tau = -1.0 + 2.0 * s
        out[j] = min(c * d / max(tau, 1e-9), c * d)
    return out


# ---------------------------------------------------------------------------
# BWM clássico (modelo linear de Rezaei, 2016) para comparação
# ---------------------------------------------------------------------------


def classic_bwm_linear(ab: np.ndarray, aw: np.ndarray) -> tuple[np.ndarray, float]:
    """
    min ξ  s.a. |w_B − a_Bj·w_j| ≤ ξ ,  |w_j − a_jW·w_W| ≤ ξ ,  Σw = 1 , w ≥ 0.
    Devolve (pesos, ξ^L). ξ^L próximo de 0 indica elevada consistência.
    """
    ab = np.asarray(ab, float)
    aw = np.asarray(aw, float)
    n = len(ab)
    b = int(np.argmin(ab))  # melhor critério: a_BB = 1
    wst = int(np.argmin(aw))  # pior critério: a_WW = 1
    c = np.zeros(n + 1)
    c[-1] = 1.0
    A, rhs = [], []
    for j in range(n):
        # w_B − a_Bj w_j − ξ ≤ 0   e   −w_B + a_Bj w_j − ξ ≤ 0
        r = np.zeros(n + 1)
        r[b] += 1
        r[j] -= ab[j]
        r[-1] = -1
        A.append(r.copy())
        rhs.append(0)
        r[:n] *= -1
        A.append(r)
        rhs.append(0)
        # w_j − a_jW w_W − ξ ≤ 0   e   −w_j + a_jW w_W − ξ ≤ 0
        r = np.zeros(n + 1)
        r[j] += 1
        r[wst] -= aw[j]
        r[-1] = -1
        A.append(r.copy())
        rhs.append(0)
        r[:n] *= -1
        A.append(r)
        rhs.append(0)
    Aeq = np.zeros((1, n + 1))
    Aeq[0, :n] = 1
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(rhs), A_eq=Aeq, b_eq=[1],
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        return np.full(n, np.nan), np.nan
    return res.x[:n], float(res.x[-1])


# ---------------------------------------------------------------------------
# Rácio de consistência clássico (modelo não linear de Rezaei, 2015)
# ---------------------------------------------------------------------------

# Índice de consistência CI(a_BW) — Rezaei (2015), Tabela 3.
CONSISTENCY_INDEX = {1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23}


def _feasible(ab, aw, b, wst, xi):
    n = len(ab)
    A, rhs = [], []
    for j in range(n):
        for sign in (1, -1):
            r = np.zeros(n)
            r[b] += sign
            r[j] -= sign * ab[j] + xi
            A.append(r)
            rhs.append(0)
            r = np.zeros(n)
            r[j] += sign
            r[wst] -= sign * aw[j] + xi
            A.append(r)
            rhs.append(0)
    res = linprog(np.zeros(n), A_ub=np.array(A), b_ub=np.array(rhs), A_eq=np.ones((1, n)), b_eq=[1],
                  bounds=[(1e-9, None)] * n, method="highs")
    return res.status == 0


def classic_cr(ab: np.ndarray, aw: np.ndarray) -> tuple[float, float]:
    """
    ξ* do modelo não linear  min ξ  s.a. |w_B/w_j − a_Bj| ≤ ξ ,  |w_j/w_W − a_jW| ≤ ξ ,
    obtido por bissecção com teste de viabilidade linear. Devolve (ξ*, CR = ξ*/CI(a_BW)).
    """
    ab = np.asarray(ab, float)
    aw = np.asarray(aw, float)
    b, wst = int(np.argmin(ab)), int(np.argmin(aw))
    a_bw = int(round(min(max(ab[wst], 1), 9)))
    lo, hi = 0.0, float(max(ab.max(), aw.max()))
    if _feasible(ab, aw, b, wst, 0.0):
        return 0.0, 0.0
    for _ in range(45):
        mid = (lo + hi) / 2
        if _feasible(ab, aw, b, wst, mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    ci = CONSISTENCY_INDEX[a_bw]
    return hi, (hi / ci if ci > 0 else np.nan)


# ---------------------------------------------------------------------------
# Consistência baseada nos inputs (Liang, Brunelli & Rezaei, 2020)
# ---------------------------------------------------------------------------

# Limiares de CR^I por (a_BW, n). Linhas: a_BW = 3..9 ; colunas: n = 3..9.
# Valores de referência de Liang et al. (2020) — confirmar com a publicação original
# antes de usar em trabalhos formais.
_CRI_THRESH = {
    3: [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],
    4: [0.1121, 0.1529, 0.1898, 0.2206, 0.2527, 0.2577, 0.2683],
    5: [0.1354, 0.1994, 0.2306, 0.2546, 0.2716, 0.2844, 0.2960],
    6: [0.1330, 0.1990, 0.2643, 0.3044, 0.3144, 0.3221, 0.3262],
    7: [0.1294, 0.2457, 0.2819, 0.3029, 0.3144, 0.3251, 0.3403],
    8: [0.1309, 0.2521, 0.2958, 0.3154, 0.3408, 0.3620, 0.3657],
    9: [0.1359, 0.2681, 0.3062, 0.3337, 0.3517, 0.3620, 0.3662],
}


def input_based_cr(ab: np.ndarray, aw: np.ndarray) -> tuple[float, float | None]:
    """
    CR^I = max_j |a_Bj·a_jW − a_BW| / (a_BW² − a_BW).
    Devolve (CR^I, limiar) — limiar None quando a_BW ≤ 2 (sem tabela aplicável).
    """
    ab = np.asarray(ab, float)
    aw = np.asarray(aw, float)
    wst = int(np.argmin(aw))
    a_bw = ab[wst]
    n = len(ab)
    if a_bw <= 1:
        return 0.0, None
    cr = float(np.max(np.abs(ab * aw - a_bw)) / (a_bw ** 2 - a_bw))
    key = int(round(min(max(a_bw, 3), 9)))
    col = min(max(n, 3), 9) - 3
    thr = _CRI_THRESH[key][col] if a_bw >= 3 else None
    return cr, thr


# ---------------------------------------------------------------------------
# Avaliação de alternativas com propagação da incerteza
# ---------------------------------------------------------------------------


def score_alternatives(perf: np.ndarray, is_benefit: list[bool], w_samples: np.ndarray):
    """
    Normalização linear (máx-mín) e soma ponderada (SAW), aplicada a cada amostra de w*.
    perf: (m alternativas, n critérios). Devolve (pontuações (S, m), matriz normalizada).
    """
    perf = np.asarray(perf, float)
    norm = np.empty_like(perf)
    for j in range(perf.shape[1]):
        col = perf[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        rng_ = hi - lo if hi > lo else 1.0
        norm[:, j] = (col - lo) / rng_ if is_benefit[j] else (hi - col) / rng_
    scores = w_samples @ norm.T
    return scores, norm
