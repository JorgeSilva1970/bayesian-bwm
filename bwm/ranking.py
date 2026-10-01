"""
Ordenação de alternativas com os pesos do Bayesian BWM.

Cada método é aplicado a TODAS as amostras a posteriori dos pesos (Monte Carlo), o que dá
não só uma pontuação média mas também a probabilidade de cada alternativa ficar em cada posição.

- SAW (Simple Additive Weighting): normalização máx-mín e soma ponderada.
- TOPSIS (Hwang & Yoon, 1981): proximidade relativa à solução ideal positiva e distância à negativa.
- VIKOR (Opricovic & Tzeng, 2004): solução de compromisso; índice Q, menor é melhor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ["TOPSIS", "VIKOR", "SAW"]
METHOD_INFO = {
    "SAW": "Soma ponderada dos desempenhos normalizados (0-1). Simples e transparente, mas totalmente "
           "compensatória: um mau desempenho pode ser compensado por outro muito bom.",
    "TOPSIS": "Escolhe a alternativa mais próxima da solução ideal (melhor valor em cada critério) e mais "
              "afastada da anti-ideal. Pontuação C entre 0 e 1: quanto maior, melhor.",
    "VIKOR": "Procura uma solução de compromisso entre a utilidade do grupo (S) e o arrependimento "
             "individual máximo (R). Índice Q entre 0 e 1: quanto MENOR, melhor.",
}


def _prep(perf, is_benefit):
    X = np.asarray(perf, float)
    b = np.asarray(is_benefit, bool)
    return X, b


def saw_scores(perf, is_benefit, W):
    X, b = _prep(perf, is_benefit)
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    N = np.where(b, (X - lo) / rng, (hi - X) / rng)
    return W @ N.T, N


def topsis_scores(perf, is_benefit, W):
    """Devolve C (S, m): proximidade relativa. W: (S, n)."""
    X, b = _prep(perf, is_benefit)
    norm = np.sqrt((X ** 2).sum(axis=0))
    R = X / np.where(norm > 0, norm, 1.0)
    V = W[:, None, :] * R[None, :, :]                  # (S, m, n)
    vmax, vmin = V.max(axis=1), V.min(axis=1)          # (S, n)
    pos = np.where(b, vmax, vmin)
    neg = np.where(b, vmin, vmax)
    dpos = np.sqrt(((V - pos[:, None, :]) ** 2).sum(axis=2))
    dneg = np.sqrt(((V - neg[:, None, :]) ** 2).sum(axis=2))
    tot = dpos + dneg
    return np.where(tot > 0, dneg / np.where(tot > 0, tot, 1.0), 0.5), R


def vikor_scores(perf, is_benefit, W, v=0.5):
    """Devolve (Q, S, R) com forma (amostras, alternativas). Menor Q é melhor."""
    X, b = _prep(perf, is_benefit)
    best = np.where(b, X.max(axis=0), X.min(axis=0))
    worst = np.where(b, X.min(axis=0), X.max(axis=0))
    den = np.where(best != worst, best - worst, 1.0)
    D = (best - X) / den                               # (m, n), 0 = melhor
    D = np.where(best != worst, D, 0.0)
    WD = W[:, None, :] * D[None, :, :]                 # (S, m, n)
    S = WD.sum(axis=2)
    R = WD.max(axis=2)

    def _scale(M):
        lo, hi = M.min(axis=1, keepdims=True), M.max(axis=1, keepdims=True)
        return np.where(hi > lo, (M - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)

    Q = v * _scale(S) + (1 - v) * _scale(R)
    return Q, S, R


def method_scores(method, perf, is_benefit, W, v=0.5):
    """Pontuação orientada para 'maior é melhor' (VIKOR: 1 − Q) e a métrica original."""
    if method == "SAW":
        sc, _ = saw_scores(perf, is_benefit, W)
        return sc, sc
    if method == "TOPSIS":
        sc, _ = topsis_scores(perf, is_benefit, W)
        return sc, sc
    if method == "VIKOR":
        Q, _, _ = vikor_scores(perf, is_benefit, W, v)
        return 1 - Q, Q
    raise ValueError(method)


def rank_probabilities(scores):
    m = scores.shape[1]
    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)
    return np.stack([(ranks == r).mean(axis=0) for r in range(m)], axis=1), ranks + 1


def evaluate(alt: pd.DataFrame, types: dict, criteria: list[str], W: np.ndarray, v: float = 0.5) -> dict:
    """Aplica os três métodos. alt: índice = alternativas, colunas = critérios."""
    alt = alt[criteria].apply(pd.to_numeric, errors="coerce")
    if alt.isna().any().any():
        raise ValueError("A tabela de alternativas tem células vazias ou não numéricas.")
    if len(alt) < 2:
        raise ValueError("São precisas pelo menos 2 alternativas.")
    is_benefit = [types.get(c, "Benefício") == "Benefício" for c in criteria]
    names = [str(i) for i in alt.index]
    X = alt.values
    out = {"names": names, "is_benefit": is_benefit, "X": X, "criteria": criteria, "v": v, "methods": {}}
    for mth in METHODS:
        sc, raw = method_scores(mth, X, is_benefit, W, v)
        rp, ranks = rank_probabilities(sc)
        metric = "Q (menor é melhor)" if mth == "VIKOR" else ("C (proximidade)" if mth == "TOPSIS" else "Pontuação")
        tab = pd.DataFrame({
            "Alternativa": names,
            f"{metric} média": raw.mean(axis=0),
            "IC 95% inf.": np.percentile(raw, 2.5, axis=0),
            "IC 95% sup.": np.percentile(raw, 97.5, axis=0),
            "P(1.º lugar)": rp[:, 0],
            "Posição esperada": ranks.mean(axis=0),
        })
        tab = tab.sort_values(f"{metric} média", ascending=(mth == "VIKOR")).reset_index(drop=True)
        tab.insert(0, "Posição", range(1, len(tab) + 1))
        out["methods"][mth] = {
            "scores": sc, "raw": raw, "table": tab, "metric": metric,
            "rank_prob": pd.DataFrame(rp, index=names, columns=[f"{r + 1}.º" for r in range(len(names))]),
        }
    out["comparison"] = comparison_table(out)
    out["vikor_check"] = vikor_conditions(X, is_benefit, W.mean(axis=0), names, v)
    return out


def comparison_table(ev: dict) -> pd.DataFrame:
    rows = {}
    for mth in METHODS:
        tab = ev["methods"][mth]["table"].set_index("Alternativa")
        rows[f"Posição {mth}"] = tab["Posição"]
        rows[f"P(1.º) {mth}"] = tab["P(1.º lugar)"]
    df = pd.DataFrame(rows).loc[ev["names"]]
    pos_cols = [f"Posição {m}" for m in METHODS]
    df["Posição média"] = df[pos_cols].mean(axis=1)
    df["Métodos concordam"] = df[pos_cols].nunique(axis=1).eq(1).map({True: "Sim", False: "Não"})
    return df.sort_values("Posição média").reset_index().rename(columns={"index": "Alternativa"})


def spearman_matrix(ev: dict) -> pd.DataFrame:
    pos = {m: ev["methods"][m]["table"].set_index("Alternativa").loc[ev["names"], "Posição"].values
           for m in METHODS}
    n = len(ev["names"])
    M = np.ones((3, 3))
    for i, a in enumerate(METHODS):
        for j, b in enumerate(METHODS):
            d = pos[a] - pos[b]
            M[i, j] = 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1)) if n > 1 else 1.0
    return pd.DataFrame(M, index=METHODS, columns=METHODS)


def vikor_conditions(X, is_benefit, w, names, v=0.5) -> dict:
    """Condições de aceitação do VIKOR (vantagem aceitável e estabilidade) com os pesos médios."""
    Q, S, R = (a[0] for a in vikor_scores(X, is_benefit, w[None, :], v))
    m = len(names)
    oq = np.argsort(Q)
    a1, a2 = oq[0], oq[1]
    dq = 1 / (m - 1)
    c1 = Q[a2] - Q[a1] >= dq
    c2 = (np.argmin(S) == a1) or (np.argmin(R) == a1)
    if c1 and c2:
        comp = [names[a1]]
        msg = f"«{names[a1]}» cumpre as duas condições do VIKOR e é a solução de compromisso."
    elif not c1:
        # conjunto de compromisso: alternativas com Q(a) − Q(a1) < DQ
        comp = [names[i] for i in oq if Q[i] - Q[a1] < dq]
        msg = ("Vantagem aceitável não cumprida: as alternativas " + ", ".join(f"«{c}»" for c in comp)
               + " formam o conjunto de soluções de compromisso.")
    else:
        comp = [names[a1], names[a2]]
        msg = (f"Estabilidade não cumprida: «{names[a1]}» e «{names[a2]}» são ambas soluções de compromisso.")
    return {"DQ": dq, "C1_vantagem": bool(c1), "C2_estabilidade": bool(c2), "compromisso": comp,
            "mensagem": msg, "Q": Q, "S": S, "R": R}


def sensitivity(method, X, is_benefit, w_mean, crit_idx, v=0.5, steps=51):
    """
    Varia o peso do critério crit_idx de 0 a 1, redistribuindo os restantes na proporção atual.
    Devolve (grelha de pesos, pontuações (steps, m), pontos onde a alternativa líder muda).
    """
    grid = np.linspace(0, 1, steps)
    others = np.delete(w_mean, crit_idx)
    others = others / others.sum() if others.sum() > 0 else np.full_like(others, 1 / len(others))
    W = np.empty((steps, len(w_mean)))
    for k, g in enumerate(grid):
        W[k] = np.insert(others * (1 - g), crit_idx, g)
    sc, raw = method_scores(method, X, is_benefit, W, v)
    leader = sc.argmax(axis=1)
    switches = [(float(grid[k]), int(leader[k - 1]), int(leader[k]))
                for k in range(1, steps) if leader[k] != leader[k - 1]]
    return grid, raw, switches
