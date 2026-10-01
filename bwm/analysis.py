"""Junta resultados do modelo em tabelas e gera a interpretação automática."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ranking as RK
from .model import BayesBWMResult, classic_bwm_linear, classic_cr, input_based_cr


def weights_table(res: BayesBWMResult, cred: float = 0.95) -> pd.DataFrame:
    s = res.summary(cred)
    R = res.rank_probabilities()
    df = pd.DataFrame({
        "Critério": res.criteria,
        "Peso médio": s["media"],
        "Desvio-padrão": s["desvio"],
        "Mediana": s["mediana"],
        f"IC {int(cred * 100)}% inf.": s["ic_inf"],
        f"IC {int(cred * 100)}% sup.": s["ic_sup"],
        "P(ser o mais importante)": R[:, 0],
    }).sort_values("Peso médio", ascending=False).reset_index(drop=True)
    df.insert(0, "Posição", range(1, len(df) + 1))
    return df


def credal_table(res: BayesBWMResult) -> pd.DataFrame:
    P = res.credal_matrix()
    return pd.DataFrame(P, index=res.criteria, columns=res.criteria)


def credal_chain(res: BayesBWMResult) -> list[tuple[str, str, float]]:
    """Ranking ordenado pela média, com a confiança entre posições consecutivas."""
    mean = res.w_star.mean(axis=0)
    order = np.argsort(-mean)
    P = res.credal_matrix()
    return [(res.criteria[a], res.criteria[b], float(P[a, b])) for a, b in zip(order[:-1], order[1:])]


def consistency_table(bo: pd.DataFrame, ow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    crit = list(bo.columns)
    for dm in bo.index:
        ab, aw = bo.loc[dm].values.astype(float), ow.loc[dm].values.astype(float)
        cr, thr = input_based_cr(ab, aw)
        w, xi = classic_bwm_linear(ab, aw)
        xi_star, cr_classic = classic_cr(ab, aw)
        if thr is None:
            estado = "Sem limiar (a_BW ≤ 2)"
        else:
            estado = "Aceitável" if cr <= thr else "Rever julgamentos"
        rows.append({
            "Decisor": dm, "Melhor": crit[int(np.argmin(ab))], "Pior": crit[int(np.argmin(aw))],
            "a_BW": ab[int(np.argmin(aw))], "CR input-based": cr,
            "Limiar": thr if thr is not None else np.nan, "Avaliação": estado,
            "CR clássico": cr_classic, "ξ* (não linear)": xi_star, "ξ (BWM linear)": xi,
        })
    return pd.DataFrame(rows)


def classic_weights_table(bo: pd.DataFrame, ow: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for dm in bo.index:
        w, _ = classic_bwm_linear(bo.loc[dm].values.astype(float), ow.loc[dm].values.astype(float))
        out[dm] = w
    df = pd.DataFrame(out, index=bo.columns).T
    df.loc["Média aritmética"] = df.mean(axis=0)
    return df


def individual_table(res: BayesBWMResult) -> pd.DataFrame:
    df = pd.DataFrame(res.individual_means(), index=res.decision_makers, columns=res.criteria)
    df.loc["Grupo (w*)"] = res.w_star.mean(axis=0)
    return df


def diagnostics_table(res: BayesBWMResult) -> pd.DataFrame:
    d = res.diagnostics()
    return pd.DataFrame({"Critério": res.criteria, "R-hat": d["rhat"], "ESS": d["ess"]})


def dm_distance(res: BayesBWMResult) -> pd.DataFrame:
    """Distância de cada decisor ao consenso (divergência de Jensen-Shannon, base 2, 0-1)."""
    ws = res.w_star.mean(axis=0)
    rows = []
    for k, dm in enumerate(res.decision_makers):
        wk = res.w_ind[:, k, :].mean(axis=0)
        m = 0.5 * (wk + ws)
        js = 0.5 * np.sum(wk * np.log2(wk / m)) + 0.5 * np.sum(ws * np.log2(ws / m))
        rows.append({"Decisor": dm, "Distância ao consenso (JS)": float(js),
                     "Critério que mais valoriza": res.criteria[int(np.argmax(wk))]})
    return pd.DataFrame(rows).sort_values("Distância ao consenso (JS)", ascending=False)


def alternatives_analysis(res: BayesBWMResult, alt: pd.DataFrame, types: dict[str, str], v: float = 0.5) -> dict:
    """SAW, TOPSIS e VIKOR aplicados a todas as amostras a posteriori dos pesos."""
    return RK.evaluate(alt, types, res.criteria, res.w_star, v)


# ---------------------------------------------------------------------------
# Interpretação automática
# ---------------------------------------------------------------------------


def _pct(p: float) -> str:
    if p > 0.995:
        return "mais de 99%"
    if p < 0.005:
        return "menos de 1%"
    return f"{p:.0%}"


def interpretation(res: BayesBWMResult, cons: pd.DataFrame, alt_res: dict | None = None,
                   method: str = "TOPSIS") -> list[str]:
    wt = weights_table(res)
    chain = credal_chain(res)
    diag = res.diagnostics()
    g_med = float(np.median(res.gamma))
    top = wt.iloc[0]
    txt = []
    txt.append(
        f"O critério mais importante para o grupo é «{top['Critério']}», com peso médio de "
        f"{top['Peso médio']:.3f} (intervalo de credibilidade 95%: {top.iloc[5]:.3f} a {top.iloc[6]:.3f}). "
        f"A probabilidade de ser efetivamente o mais importante é de {_pct(top['P(ser o mais importante)'])}."
    )
    bottom = wt.iloc[-1]
    txt.append(
        f"O critério menos relevante é «{bottom['Critério']}» (peso médio {bottom['Peso médio']:.3f}). "
        f"Os três primeiros critérios concentram {wt['Peso médio'].head(3).sum():.0%} do peso total."
    )
    weak = [(a, b, p) for a, b, p in chain if p < 0.65]
    strong = [(a, b, p) for a, b, p in chain if p >= 0.9]
    if strong:
        txt.append("Precedências com confiança elevada: " + "; ".join(
            f"«{a}» > «{b}» ({p:.2f})" for a, b, p in strong) + ".")
    if weak:
        txt.append("Precedências incertas (a ordem pode inverter-se): " + "; ".join(
            f"«{a}» vs «{b}» ({p:.2f})" for a, b, p in weak)
            + ". Nestes pares, as decisões não devem depender da ordem exata.")
    if g_med > 200:
        agr = "elevada concordância"
    elif g_med > 40:
        agr = "concordância moderada"
    else:
        agr = "divergência considerável"
    txt.append(
        f"A concentração γ (mediana {g_med:.1f}) indica {agr} entre os decisores: quanto maior γ, "
        "mais próximos estão os pesos individuais do consenso do grupo."
    )
    bad = cons[cons["Avaliação"] == "Rever julgamentos"]
    if len(bad):
        txt.append("Decisores com julgamentos pouco consistentes (convém rever as comparações): "
                   + ", ".join(bad["Decisor"]) + ".")
    else:
        txt.append("Todos os decisores apresentam consistência aceitável ou sem limiar aplicável.")
    rmax = float(np.nanmax(diag["rhat"]))
    emin = float(np.nanmin(diag["ess"]))
    if rmax < 1.01 and emin > 400:
        txt.append(f"Diagnóstico MCMC: convergência adequada (R-hat máx. {rmax:.3f}; ESS mín. {emin:.0f}).")
    else:
        txt.append(f"Diagnóstico MCMC: R-hat máx. {rmax:.3f} e ESS mín. {emin:.0f}. "
                   "Recomenda-se aumentar o número de iterações para resultados mais estáveis.")
    if alt_res is not None:
        mres = alt_res["methods"][method]
        t = mres["table"]
        metric = mres["metric"].split(" ")[0]
        a1 = t.iloc[0]
        s = (f"Com o {method}, «{a1['Alternativa']}» é a melhor alternativa ({metric} médio "
             f"{a1.iloc[2]:.3f}) e fica em 1.º lugar em {_pct(a1['P(1.º lugar)'])} das amostras de pesos.")
        if len(t) > 1:
            a2 = t.iloc[1]
            s += (f" Segue-se «{a2['Alternativa']}» ({metric} {a2.iloc[2]:.3f}; "
                  f"1.º lugar em {_pct(a2['P(1.º lugar)'])} das amostras).")
        txt.append(s)
        comp = alt_res["comparison"]
        leaders = {m: alt_res["methods"][m]["table"].iloc[0]["Alternativa"] for m in RK.METHODS}
        if len(set(leaders.values())) == 1:
            txt.append(f"SAW, TOPSIS e VIKOR concordam na alternativa líder («{a1['Alternativa']}»), "
                       "o que reforça a robustez da recomendação.")
        else:
            txt.append("Os métodos não concordam na alternativa líder: " + "; ".join(
                f"{m} → «{a}»" for m, a in leaders.items())
                + ". A escolha depende do grau de compensação entre critérios que se aceita.")
        n_agree = int((comp["Métodos concordam"] == "Sim").sum())
        txt.append(f"As três técnicas atribuem a mesma posição a {n_agree} de {len(comp)} alternativas.")
        txt.append("VIKOR: " + alt_res["vikor_check"]["mensagem"])
    return txt
