"""Gráficos: Plotly para a app/relatório HTML, Matplotlib para o relatório Word."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

PINE = "#2E5E4E"
PINE_LIGHT = "#8FB8A8"
AMBER = "#C98A1B"
INK = "#1B2A26"
SCALE = [[0, "#F2F5F3"], [0.5, "#8FB8A8"], [1, "#1F4437"]]
FONT = dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color=INK)


def _layout(fig, title=None, height=420):
    fig.update_layout(title=title, height=height, font=FONT, margin=dict(l=10, r=10, t=50, b=10),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------


def weights_bar(criteria, summ):
    order = np.argsort(summ["media"])
    c = [criteria[i] for i in order]
    m = summ["media"][order]
    fig = go.Figure(go.Bar(
        x=m, y=c, orientation="h", marker_color=PINE,
        error_x=dict(type="data", symmetric=False, array=summ["ic_sup"][order] - m,
                     arrayminus=m - summ["ic_inf"][order], color=AMBER, thickness=1.5),
        text=[f"{v:.3f}" for v in m], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white"),
        hovertemplate="%{y}<br>peso médio %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Peso agregado w* (barra = intervalo de credibilidade)", gridcolor="#E6ECE9")
    return _layout(fig, "Pesos agregados do grupo", height=90 + 45 * len(criteria))


def credal_heatmap(criteria, P):
    txt = np.where(np.isnan(P), "", np.vectorize(lambda v: f"{v:.2f}")(np.nan_to_num(P)))
    fig = go.Figure(go.Heatmap(
        z=P, x=criteria, y=criteria, colorscale=SCALE, zmin=0, zmax=1, text=txt,
        texttemplate="%{text}", hovertemplate="P(%{y} > %{x}) = %{z:.2f}<extra></extra>",
        colorbar=dict(title="Confiança"),
    ))
    fig.update_yaxes(autorange="reversed", title="Critério da linha")
    fig.update_xaxes(title="é mais importante do que o critério da coluna", tickangle=-30)
    return _layout(fig, "Matriz de ranking credal — P(linha > coluna)", height=160 + 50 * len(criteria))


def rank_prob_heatmap(criteria, R):
    n = len(criteria)
    fig = go.Figure(go.Heatmap(
        z=R, x=[f"{r + 1}.º" for r in range(n)], y=criteria, colorscale=SCALE, zmin=0, zmax=1,
        text=np.vectorize(lambda v: f"{v:.2f}")(R), texttemplate="%{text}",
        hovertemplate="%{y} na posição %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Posição no ranking")
    return _layout(fig, "Probabilidade de cada critério ocupar cada posição", height=160 + 45 * n)


def posterior_violin(criteria, W):
    fig = go.Figure()
    order = np.argsort(-W.mean(axis=0))
    for i in order:
        fig.add_trace(go.Violin(y=W[:, i], name=criteria[i], line_color=PINE, fillcolor=PINE_LIGHT,
                                opacity=0.8, meanline_visible=True, points=False, showlegend=False))
    fig.update_yaxes(title="Peso w*", gridcolor="#E6ECE9")
    return _layout(fig, "Distribuição a posteriori dos pesos agregados")


def individual_heatmap(criteria, dms, Wind, wstar_mean):
    z = np.vstack([Wind, wstar_mean])
    y = list(dms) + ["▶ Grupo (w*)"]
    fig = go.Figure(go.Heatmap(
        z=z, x=criteria, y=y, colorscale=SCALE, text=np.vectorize(lambda v: f"{v:.3f}")(z),
        texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-30)
    return _layout(fig, "Pesos individuais estimados vs. pesos do grupo", height=160 + 40 * len(y))


def gamma_hist(g):
    fig = go.Figure(go.Histogram(x=g, nbinsx=60, marker_color=PINE))
    fig.update_xaxes(title="γ (concentração)", type="log")
    fig.update_yaxes(title="Frequência")
    return _layout(fig, "Grau de concordância entre decisores (γ)", height=320)


def trace_plot(W, chains, draws, idx, name):
    fig = go.Figure()
    for c in range(chains):
        seg = W[c * draws:(c + 1) * draws, idx]
        fig.add_trace(go.Scatter(y=seg, mode="lines", name=f"Cadeia {c + 1}", line=dict(width=0.8)))
    fig.update_xaxes(title="Iteração (após aquecimento)")
    fig.update_yaxes(title=f"w* — {name}")
    return _layout(fig, f"Traço das cadeias MCMC — {name}", height=320)


def alternatives_bar(names, raw, metric="Pontuação", lower_better=False, method=""):
    m = raw.mean(axis=0)
    lo, hi = np.percentile(raw, 2.5, axis=0), np.percentile(raw, 97.5, axis=0)
    order = np.argsort(-m) if lower_better else np.argsort(m)
    fig = go.Figure(go.Bar(
        x=m[order], y=[names[i] for i in order], orientation="h", marker_color=PINE,
        error_x=dict(type="data", symmetric=False, array=np.clip(hi - m, 0, None)[order],
                     arrayminus=np.clip(m - lo, 0, None)[order], color=AMBER),
        text=[f"{v:.3f}" for v in m[order]], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white"),
    ))
    fig.update_xaxes(title=f"{metric} — intervalo de credibilidade 95%")
    return _layout(fig, f"{method}: pontuação das alternativas (melhor no topo)", height=90 + 50 * len(names))


def method_first_prob(ev):
    fig = go.Figure()
    colors = {"TOPSIS": PINE, "VIKOR": AMBER, "SAW": PINE_LIGHT}
    for mth, d in ev["methods"].items():
        rp = d["rank_prob"].iloc[:, 0]
        fig.add_trace(go.Bar(name=mth, x=list(rp.index), y=rp.values, marker_color=colors.get(mth),
                             hovertemplate=f"{mth}<br>%{{x}}: %{{y:.0%}}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="P(1.º lugar)", tickformat=".0%", range=[0, 1], gridcolor="#E6ECE9")
    return _layout(fig, "Probabilidade de 1.º lugar por método", height=380)


def sensitivity_chart(grid, raw, names, crit_name, current, metric, switches):
    fig = go.Figure()
    for i, n in enumerate(names):
        fig.add_trace(go.Scatter(x=grid, y=raw[:, i], mode="lines", name=n, line=dict(width=2)))
    fig.add_vline(x=current, line_dash="dash", line_color=INK,
                  annotation_text="peso atual", annotation_position="top")
    for g, _, _ in switches:
        fig.add_vline(x=g, line_dash="dot", line_color=AMBER)
    fig.update_xaxes(title=f"Peso atribuído a «{crit_name}» (os restantes mantêm as proporções)", range=[0, 1])
    fig.update_yaxes(title=metric, gridcolor="#E6ECE9")
    return _layout(fig, f"Sensibilidade ao peso de «{crit_name}»", height=420)


# ---------------------------------------------------------------------------
# Grafo de ranking credal (DOT para st.graphviz_chart)
# ---------------------------------------------------------------------------


def credal_edges(criteria, P, mean, threshold=0.5):
    """Arestas i→j com P(i>j) ≥ limiar, após redução transitiva (mantém o grafo legível)."""
    n = len(criteria)
    A = (np.nan_to_num(P) >= threshold) & (np.nan_to_num(P) > 0.5)
    red = A.copy()
    for i in range(n):
        for j in range(n):
            if A[i, j]:
                for k in range(n):
                    if k not in (i, j) and A[i, k] and A[k, j]:
                        red[i, j] = False
                        break
    return [(i, j, P[i, j]) for i in range(n) for j in range(n) if red[i, j]]


def credal_dot(criteria, P, mean, threshold=0.5):
    edges = credal_edges(criteria, P, mean, threshold)
    lines = ['digraph G {', 'rankdir=LR; bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fillcolor="#EEF2F0", color="#2E5E4E", '
             'fontname="Helvetica", fontcolor="#1B2A26"];',
             'edge [color="#2E5E4E", fontname="Helvetica", fontsize=11, fontcolor="#C98A1B"];']
    for i, c in enumerate(criteria):
        lines.append(f'n{i} [label="{c}\\n{mean[i]:.3f}"];')
    for i, j, p in edges:
        lines.append(f'n{i} -> n{j} [label="{p:.2f}", penwidth={1 + 2 * (p - 0.5) * 2:.1f}];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib (imagens PNG para o relatório Word)
# ---------------------------------------------------------------------------


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def mpl_weights(criteria, summ) -> bytes:
    order = np.argsort(summ["media"])
    fig, ax = plt.subplots(figsize=(7, 0.5 + 0.45 * len(criteria)))
    m = summ["media"][order]
    ax.barh([criteria[i] for i in order], m, color=PINE,
            xerr=[m - summ["ic_inf"][order], summ["ic_sup"][order] - m], ecolor=AMBER, capsize=3)
    for y, v in enumerate(m):
        ax.text(v / 2, y, f"{v:.3f}", va="center", ha="center", color="white", fontsize=8)
    ax.set_xlabel("Peso agregado w* (intervalo de credibilidade 95%)")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def mpl_heatmap(M, xlabels, ylabels, title, fmt="{:.2f}") -> bytes:
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(xlabels), 0.8 + 0.5 * len(ylabels)))
    im = ax.imshow(np.nan_to_num(M, nan=np.nan), cmap="Greens", vmin=0,
                   vmax=np.nanmax(M) if np.nanmax(M) > 1 else 1)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * np.nanmax(M) else INK)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    return _png(fig)


def mpl_alternatives(names, raw, metric="Pontuação", lower_better=False) -> bytes:
    m = raw.mean(axis=0)
    lo, hi = np.percentile(raw, 2.5, axis=0), np.percentile(raw, 97.5, axis=0)
    order = np.argsort(-m) if lower_better else np.argsort(m)
    fig, ax = plt.subplots(figsize=(7, 0.6 + 0.5 * len(names)))
    ax.barh([names[i] for i in order], m[order], color=PINE,
            xerr=[np.clip(m - lo, 0, None)[order], np.clip(hi - m, 0, None)[order]], ecolor=AMBER, capsize=3)
    ax.set_xlabel(f"{metric} (melhor no topo)")
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def mpl_method_first(ev) -> bytes:
    names = ev["names"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    cols = {"TOPSIS": PINE, "VIKOR": AMBER, "SAW": PINE_LIGHT}
    for k, (mth, d) in enumerate(ev["methods"].items()):
        ax.bar(x + (k - 1) * 0.27, d["rank_prob"].iloc[:, 0].values, 0.27, label=mth, color=cols[mth])
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("P(1.º lugar)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)
