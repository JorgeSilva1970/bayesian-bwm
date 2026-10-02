"""
Bayesian BWM Studio — aplicação Streamlit para o Bayesian Best-Worst Method.

Executar localmente:
    python -m streamlit run app.py
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

import os

from bwm import ai as AI
from bwm import analysis as A
from bwm import charts as C
from bwm import data as D
from bwm import learn as L
from bwm import ranking as RK
from bwm import report as R
from bwm import templates as T
from bwm.model import bayesian_bwm

st.set_page_config(page_title="Bayesian BWM Studio", page_icon="⚖️", layout="wide")
ss = st.session_state

SCALE_TABLE = pd.DataFrame({
    "Valor": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Significado": ["Igual importância", "Entre igual e moderada", "Moderadamente mais importante",
                    "Entre moderada e forte", "Fortemente mais importante", "Entre forte e muito forte",
                    "Muito fortemente mais importante", "Entre muito forte e extrema",
                    "Extremamente mais importante"],
})

# ---------------------------------------------------------------------------
# Estado da sessão
# ---------------------------------------------------------------------------


def _bump(name: str):
    ss[f"{name}_ver"] = ss.get(f"{name}_ver", 0) + 1


def _empty_matrix(dms, criteria):
    return pd.DataFrame(np.nan, index=pd.Index(dms, name="Decisor"), columns=criteria, dtype=float)


def set_inputs(criteria_df, dms_df, bo=None, ow=None, alt=None, title=None, context=None, sector=None):
    criteria_df = criteria_df.copy()
    for col in D.CRIT_COLS:
        if col not in criteria_df:
            criteria_df[col] = "" if col != "Tipo" else D.BENEFICIO
    ss.crit_base = criteria_df[D.CRIT_COLS].fillna("").reset_index(drop=True)
    ss.dm_base = dms_df.reset_index(drop=True)
    crit = ss.crit_base["Critério"].tolist()
    dms = ss.dm_base["Decisor"].tolist()
    ss.bo_base = (bo if bo is not None else _empty_matrix(dms, crit)).astype(float)
    ss.ow_base = (ow if ow is not None else _empty_matrix(dms, crit)).astype(float)
    ss.bo_base.index.name = ss.ow_base.index.name = "Decisor"
    ss.bo_last, ss.ow_last = ss.bo_base, ss.ow_base
    ss.mat_sig = (tuple(crit), tuple(dms))
    ss.alt_base = alt if alt is not None else pd.DataFrame(
        columns=crit, index=pd.Index([], name="Alternativa"), dtype=float)
    ss.alt_on = alt is not None and len(alt) > 0
    if title is not None:
        ss.dec_title = title
    if context is not None:
        ss.dec_context = context
    if sector is not None:
        ss.sector_label = sector
    ss.pop("ai_reading", None)
    for n in ("crit", "dm", "mat", "alt"):
        _bump(n)
    ss.pop("result", None)


def load_example():
    ex = D.example_case(ss.get("example_name", D.DEFAULT_EXAMPLE))
    ss.industry = ex["industry"]
    set_inputs(ex["criteria"], ex["dms"], ex["bo"], ex["ow"], ex["alt"], ex["title"], ex["context"],
               ex["sector"])


def on_industry_change():
    ind = ss.industry
    ss.source = "Introdução manual"
    dms = ss.get("dm_base", D.default_dm_df())
    set_inputs(D.industry_criteria_df(ind), dms, title=D.INDUSTRIES[ind]["decisao"], context="",
               sector="" if ind == "Personalizado" else ind)


def on_source_change():
    if ss.source == "Caso exemplo":
        load_example()


if "crit_base" not in ss:
    ss.source = "Caso exemplo"
    ss.example_name = D.DEFAULT_EXAMPLE
    ss.cred = 0.95
    ss.alt_method = "TOPSIS"
    ss.vikor_v = 0.5
    load_example()

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚖️ Bayesian BWM Studio")
    st.caption("Pesos de critérios para decisões em grupo, com incerteza quantificada.")

    st.selectbox("Setor de atividade", list(D.INDUSTRIES), key="industry", on_change=on_industry_change,
                 help="Ao mudar de setor, a lista de critérios é substituída pelos critérios sugeridos "
                      "para esse setor. Pode depois editá-los, acrescentar ou apagar.")
    st.caption(f"Decisão típica: {D.INDUSTRIES[ss.industry]['decisao']}")

    st.radio("Origem dos dados", ["Caso exemplo", "Introdução manual", "Carregar ficheiro"],
             key="source", on_change=on_source_change,
             help="Caso exemplo: dados de demonstração prontos a correr. Manual: preenche as tabelas "
                  "nos separadores. Ficheiro: CSV ou Excel no formato do modelo descarregável.")

    if ss.source == "Caso exemplo":
        st.selectbox("Caso exemplo", list(D.EXAMPLES), key="example_name", on_change=load_example,
                     help="Casos ilustrativos prontos a correr: automóvel, retalho, saúde e um exemplo "
                          "didático pequeno. Os dados são fictícios, construídos para fins pedagógicos.")

    if ss.source == "Carregar ficheiro":
        st.caption("Use o **modelo Excel** em «📋 Modelos e formulários» (mais abaixo): já traz as folhas, "
                   "colunas e validações corretas.")
        up = st.file_uploader("Ficheiro de dados (Excel recomendado, ou CSV)", type=["csv", "xlsx", "xls"],
                              help="Excel: folhas Projeto, Criterios, Decisores, Comparacoes e Alternativas. "
                                   "CSV: só comparações, duas linhas por decisor (Tipo = BO e OW).")
        if up is not None and ss.get("last_upload") != (up.name, up.size):
            try:
                parsed = D.read_uploaded(up)
                bo, ow = parsed["bo"], parsed["ow"]
                crit_df = parsed.get("criteria")
                if crit_df is not None:
                    missing = [c for c in bo.columns if c not in set(crit_df["Critério"])]
                    if missing:
                        st.warning(f"Critérios sem linha na folha Criterios (assumidos como Benefício): {missing}")
                    crit_df = crit_df.set_index("Critério").reindex(bo.columns).reset_index()
                    crit_df["Tipo"] = crit_df["Tipo"].fillna(D.BENEFICIO)
                else:
                    crit_df = pd.DataFrame({"Critério": bo.columns, "Tipo": D.BENEFICIO, "Unidade": "",
                                            "Descrição": ""})
                fun = {}
                if parsed.get("dms") is not None:
                    fun = dict(zip(parsed["dms"]["Decisor"], parsed["dms"]["Função"]))
                dms_df = pd.DataFrame({"Decisor": bo.index, "Função": [fun.get(d, "") for d in bo.index]})
                pm = parsed.get("meta", {})
                alt = parsed.get("alt")
                if alt is not None:
                    alt = alt.reindex(columns=bo.columns)
                    alt.index.name = "Alternativa"
                set_inputs(crit_df, dms_df, bo, ow, alt, title=pm.get("title") or f"Análise de {up.name}",
                           context=pm.get("context", ""), sector=pm.get("sector", ""))
                ss.last_upload = (up.name, up.size)
                st.success(f"Lidos {bo.shape[0]} decisores e {bo.shape[1]} critérios.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Não foi possível ler o ficheiro: {exc}")

    st.divider()
    with st.expander("⚙️ Parâmetros da estimação", expanded=False):
        st.number_input("Amostras por cadeia", 500, 20000, 3000, 500, key="draws",
                        help="Mais amostras = estimativas mais estáveis, mas mais tempo de cálculo.")
        st.number_input("Iterações de aquecimento (burn-in)", 300, 10000, 1500, 100, key="burn",
                        help="Iterações iniciais descartadas, usadas para afinar o amostrador.")
        st.slider("Número de cadeias", 1, 6, 3, key="chains",
                  help="Várias cadeias independentes permitem verificar a convergência (R-hat).")
        st.select_slider("Nível do intervalo de credibilidade", [0.80, 0.90, 0.95, 0.99], key="cred")
        st.number_input("Semente aleatória", 0, 10_000, 42, key="seed",
                        help="Fixar a semente torna os resultados reprodutíveis.")
        st.number_input("Prior γ ~ Gamma(a, b): a", 0.001, 10.0, 0.01, format="%.3f", key="g_a")
        st.number_input("Prior γ ~ Gamma(a, b): b", 0.001, 10.0, 0.01, format="%.3f", key="g_b",
                        help="Valores por omissão (0,01; 0,01) seguem Mohammadi & Rezaei (2020): prior vago.")

# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title("Bayesian Best-Worst Method")
c1, c2 = st.columns([2, 3])
with c1:
    st.text_input("Título da decisão", key="dec_title",
                  help="Aparece no topo do relatório.")
    st.text_input("Setor / organização", key="sector_label",
                  placeholder="Ex.: Seguros — gestão de sinistros de acidentes de trabalho",
                  help="Texto livre. Substitui o setor da lista no relatório e orienta a leitura para o setor "
                       "feita com IA. Útil para setores que não estão na lista.")
with c2:
    st.text_area("Contexto (opcional)", key="dec_context", height=122,
                 help="Descreva o problema de decisão. É incluído no relatório.")

tabs = st.tabs(["1. Critérios e decisores", "2. Comparações", "3. Alternativas",
                "4. Resultados", "5. Relatório", "Aprender"])

# ---------------------------------------------------------------------------
# 1. Critérios e decisores
# ---------------------------------------------------------------------------

with tabs[0]:
    st.info("Defina **o que** vai ser ponderado (critérios) e **quem** avalia (decisores). "
            "Use a linha vazia no fim de cada tabela para acrescentar e o ícone do lixo para apagar. "
            "Não há limite para o número de critérios ou decisores.", icon="ℹ️")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Critérios")
        crit_edit = st.data_editor(
            ss.crit_base, key=f"crit_{ss.crit_ver}", num_rows="dynamic", width="stretch",
            hide_index=True,
            column_config={
                "Critério": st.column_config.TextColumn(required=True, help="Nome curto do critério."),
                "Tipo": st.column_config.SelectboxColumn(
                    options=[D.BENEFICIO, D.CUSTO], required=True, default=D.BENEFICIO,
                    help="Benefício: mais é melhor. Custo: menos é melhor. Só é usado na avaliação de alternativas."),
                "Unidade": st.column_config.TextColumn(
                    help="Unidade em que as alternativas são medidas (€, dias, %, escala 1-10). "
                         "Aparece nas ajudas, tabelas e relatórios."),
                "Descrição": st.column_config.TextColumn(width="large"),
            })
        st.caption("O tipo (benefício/custo) não afeta os pesos: serve apenas para pontuar alternativas.")
    with right:
        st.subheader("Decisores")
        dm_edit = st.data_editor(
            ss.dm_base, key=f"dm_{ss.dm_ver}", num_rows="dynamic", width="stretch",
            hide_index=True,
            column_config={"Decisor": st.column_config.TextColumn(required=True),
                           "Função": st.column_config.TextColumn(help="Área ou papel (opcional).")})
        st.caption("Com um só decisor o modelo funciona, mas o valor do Bayesian BWM está em agregar vários.")


def _clean_names(series) -> list[str]:
    out, seen = [], set()
    for v in series:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        v = str(v).strip()
        if not v:
            continue
        base, i = v, 2
        while v in seen:
            v = f"{base} ({i})"
            i += 1
        seen.add(v)
        out.append(v)
    return out


criteria = _clean_names(crit_edit["Critério"])
dms = _clean_names(dm_edit["Decisor"])
crit_types = {str(r["Critério"]).strip(): r["Tipo"] for _, r in crit_edit.dropna(subset=["Critério"]).iterrows()}
crit_df_now = crit_edit.dropna(subset=["Critério"]).reset_index(drop=True)
crit_df_now["Critério"] = crit_df_now["Critério"].astype(str).str.strip()
def _txt(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


crit_units = {r["Critério"]: _txt(r.get("Unidade")) for _, r in crit_df_now.iterrows()}
dm_functions = {_txt(r["Decisor"]): _txt(r.get("Função")) for _, r in dm_edit.dropna(subset=["Decisor"]).iterrows()}


def _reshape(prev: pd.DataFrame, rows, cols) -> pd.DataFrame:
    """Reajusta uma matriz às novas listas, preservando valores (por nome ou por posição se renomeado)."""
    prev = prev.copy()
    if len(prev.columns) == len(cols) and list(prev.columns) != list(cols):
        prev.columns = cols
    if len(prev.index) == len(rows) and list(prev.index) != list(rows):
        prev.index = rows
    out = prev.reindex(index=rows, columns=cols).astype(float)
    out.index.name = prev.index.name or "Decisor"
    return out


sig = (tuple(criteria), tuple(dms))
if sig != ss.mat_sig:
    ss.bo_base = _reshape(ss.bo_last, dms, criteria)
    ss.ow_base = _reshape(ss.ow_last, dms, criteria)
    ss.mat_sig = sig
    _bump("mat")
if list(ss.alt_base.columns) != criteria:
    ss.alt_base = _reshape(ss.alt_base, list(ss.alt_base.index), criteria)
    ss.alt_base.index.name = "Alternativa"
    _bump("alt")

# ---------------------------------------------------------------------------
# 2. Comparações
# ---------------------------------------------------------------------------

num_cfg = {c: st.column_config.NumberColumn(min_value=1, max_value=9, step=1, format="%d") for c in criteria}

with tabs[1]:
    st.info(
        "Cada decisor escolhe o **melhor** (mais importante) e o **pior** (menos importante) critério e "
        "compara-os com os restantes numa escala de 1 a 9.\n\n"
        "- **Best-to-Others (BO):** quanto o melhor critério é mais importante do que cada critério. "
        "O próprio melhor critério leva **1**.\n"
        "- **Others-to-Worst (OW):** quanto cada critério é mais importante do que o pior. "
        "O próprio pior critério leva **1**.\n\n"
        "O melhor e o pior são detetados automaticamente pelo valor 1.", icon="ℹ️")
    with st.expander("Escala de 1 a 9"):
        st.dataframe(SCALE_TABLE, hide_index=True, width="stretch")

    if not criteria or not dms:
        st.warning("Defina pelo menos 2 critérios e 1 decisor no separador 1.")
    else:
        st.subheader("Best-to-Others")
        st.caption("Linha = decisor. Pergunta: «quanto o MEU melhor critério é mais importante do que este?»")
        bo_edit = st.data_editor(ss.bo_base, key=f"bo_{ss.mat_ver}", width="stretch",
                                 column_config=num_cfg)
        st.subheader("Others-to-Worst")
        st.caption("Linha = decisor. Pergunta: «quanto este critério é mais importante do que o MEU pior?»")
        ow_edit = st.data_editor(ss.ow_base, key=f"ow_{ss.mat_ver}", width="stretch",
                                 column_config=num_cfg)
        ss.bo_last, ss.ow_last = bo_edit, ow_edit

        errors, warns, vt = D.validate(bo_edit, ow_edit)
        st.subheader("Verificação")
        st.caption("Estado dos julgamentos de cada decisor. Corrija os erros antes de correr a análise.")
        if len(vt):
            if not errors:
                cons_live = A.consistency_table(bo_edit, ow_edit)
                vt = vt.merge(cons_live[["Decisor", "CR input-based", "Limiar", "Avaliação"]],
                              on="Decisor", how="left")
            st.dataframe(vt, hide_index=True, width="stretch",
                         column_config={"CR input-based": st.column_config.NumberColumn(format="%.3f"),
                                        "Limiar": st.column_config.NumberColumn(format="%.3f")})
        for e in errors:
            st.error(e)
        for w in warns:
            st.warning(w)
        if not errors and not warns:
            st.success("Dados completos e coerentes. Pode avançar para o separador 4.")

bo_cur = ss.bo_last.reindex(index=dms, columns=criteria)
ow_cur = ss.ow_last.reindex(index=dms, columns=criteria)


def _meta_base() -> dict:
    return {"title": ss.dec_title or "Análise Bayesian BWM", "industry": ss.industry,
            "sector": (ss.get("sector_label") or "").strip() or ss.industry, "context": ss.dec_context}


# ---------------------------------------------------------------------------
# Modelos e formulários (barra lateral)
# ---------------------------------------------------------------------------

with st.sidebar:
    with st.expander("📋 Modelos e formulários", expanded=ss.source == "Carregar ficheiro"):
        st.caption("Modelos com as folhas, colunas e validações certas para carregar dados de qualquer setor.")
        mb = _meta_base()
        dms_tpl = dm_edit.dropna(subset=["Decisor"])
        st.download_button(
            "Modelo Excel — configuração atual", width="stretch",
            data=T.excel_template(crit_df_now, dms_tpl, bo_cur, ow_cur, ss.alt_base, mb),
            file_name="modelo_bwm_atual.xlsx",
            help="Critérios, unidades, decisores e valores que estão agora na app. Ideal para guardar "
                 "o trabalho ou partilhar e voltar a carregar.")
        ind = ss.industry
        st.download_button(
            f"Modelo Excel em branco — {ind}", width="stretch",
            data=T.excel_template(D.industry_criteria_df(ind), D.default_dm_df(3), meta={
                "title": D.INDUSTRIES[ind]["decisao"], "sector": "" if ind == "Personalizado" else ind}),
            file_name="modelo_bwm_base.xlsx",
            help="Modelo base com os critérios sugeridos para o setor selecionado. Edite à vontade: pode "
                 "acrescentar critérios, decisores e alternativas.")
        st.download_button(
            "Questionário para decisores (Word)", width="stretch",
            data=T.questionnaire_docx(crit_df_now, mb), file_name="questionario_decisores_bwm.docx",
            help="Formulário com os critérios atuais para cada decisor preencher. As respostas transcrevem-se "
                 "para a folha Comparacoes do modelo Excel.")
        st.download_button(
            "Modelo CSV (só comparações)", width="stretch",
            data=D.template_long(criteria, dms).to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="modelo_bwm_comparacoes.csv")

# ---------------------------------------------------------------------------
# 3. Alternativas (opcional)
# ---------------------------------------------------------------------------

with tabs[2]:
    st.info("Passo opcional. Depois de obter os pesos, pode pontuar alternativas (fornecedores, projetos, "
            "localizações...). Os valores são normalizados entre 0 e 1 — critérios de **custo** são invertidos — "
            "e somados com os pesos. Como há milhares de amostras de pesos, obtém-se também a "
            "**probabilidade de cada alternativa ficar em 1.º lugar**.", icon="ℹ️")
    st.toggle("Avaliar alternativas", key="alt_on")
    if ss.alt_on:
        mc1, mc2 = st.columns([2, 1])
        with mc1:
            st.radio("Método principal de ordenação", RK.METHODS, key="alt_method", horizontal=True,
                     help="Os três métodos são sempre calculados e comparados; o principal é o usado na "
                          "interpretação, nos gráficos em destaque e no relatório.")
            st.caption(RK.METHOD_INFO[ss.alt_method])
        with mc2:
            st.slider("VIKOR: peso da estratégia de maioria (v)", 0.0, 1.0, key="vikor_v", step=0.05,
                      help="v = 0,5 equilibra utilidade do grupo (S) e arrependimento máximo (R). "
                           "v > 0,5 privilegia a maioria; v < 0,5 privilegia evitar maus desempenhos.")
        up_alt = st.file_uploader("Carregar alternativas (CSV/Excel: 1.ª coluna = nome, restantes = critérios)",
                                  type=["csv", "xlsx", "xls"], key="alt_up")
        if up_alt is not None and ss.get("last_alt_upload") != (up_alt.name, up_alt.size):
            try:
                a = D.read_alternatives(up_alt)
                miss = [c for c in criteria if c not in a.columns]
                if miss:
                    st.error(f"Faltam colunas para os critérios: {miss}")
                else:
                    ss.alt_base = a[criteria].apply(pd.to_numeric, errors="coerce")
                    ss.alt_base.index.name = "Alternativa"
                    ss.last_alt_upload = (up_alt.name, up_alt.size)
                    _bump("alt")
                    st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Não foi possível ler o ficheiro: {exc}")
        units = crit_units
        alt_cfg = {c: st.column_config.NumberColumn(
            f"{c} ({'↑' if crit_types.get(c) == D.BENEFICIO else '↓'})",
            help=f"{'Benefício: mais é melhor' if crit_types.get(c) == D.BENEFICIO else 'Custo: menos é melhor'}"
                 + (f" — unidade: {units[c]}" if c in units else "")) for c in criteria}
        alt_df = ss.alt_base.reset_index()
        alt_df.columns = ["Alternativa"] + criteria
        alt_edit = st.data_editor(alt_df, key=f"alt_{ss.alt_ver}", num_rows="dynamic",
                                  hide_index=True, width="stretch", column_config=alt_cfg)
        st.caption("↑ benefício (mais é melhor) · ↓ custo (menos é melhor). Use unidades naturais: €, dias, %, "
                   "pontuações. Acrescente alternativas na linha vazia.")
        alt_cur = alt_edit.dropna(subset=["Alternativa"]).set_index("Alternativa")
        alt_cur = alt_cur[alt_cur.index.astype(str).str.strip() != ""]
    else:
        alt_cur = None

# ---------------------------------------------------------------------------
# Aprender
# ---------------------------------------------------------------------------

with tabs[5]:
    learn_tabs = st.tabs(["Conceito", "Vantagens e limitações", "Ferramentas e vídeos", "Modelo matemático",
                          "Glossário", "Teste-se"])
    with learn_tabs[0]:
        st.markdown(L.CONCEITO)
        st.caption("Número de comparações por decisor")
        st.dataframe(pd.DataFrame(L.AHP_TABLE), hide_index=True)
    with learn_tabs[1]:
        v1, v2 = st.columns(2)
        with v1:
            st.subheader("Vantagens")
            st.markdown(L.VANTAGENS)
        with v2:
            st.subheader("Limitações")
            st.markdown(L.DESVANTAGENS)
    with learn_tabs[2]:
        st.subheader("Aplicações e ferramentas")
        st.dataframe(pd.DataFrame(L.FERRAMENTAS, columns=["Ferramenta", "Custo", "Tecnologia", "Para que serve"]),
                     hide_index=True, width="stretch")
        st.link_button("Abrir o repositório oficial (Majeed7/BayesianBWM)", "https://github.com/Majeed7/BayesianBWM")
        st.subheader("Vídeos tutoriais")
        st.caption("Pesquisas no YouTube com os termos certos: os vídeos disponíveis mudam com o tempo, por isso "
                   "as ligações abrem uma pesquisa atualizada em vez de um vídeo fixo.")
        for title, url in L.VIDEOS:
            st.markdown(f"- [{title}]({url})")
        st.info(L.NOTA_GEMINI, icon="📝")
    with learn_tabs[3]:
        st.subheader("Best-Worst Method")
        st.markdown(R.METHOD_TEXT[0])
        st.latex(r"A_B = (a_{B1}, a_{B2}, \dots, a_{Bn}), \qquad A_W = (a_{1W}, a_{2W}, \dots, a_{nW})^T")
        st.latex(r"\min \xi \;\; \text{s.a.} \;\; \left|\tfrac{w_B}{w_j} - a_{Bj}\right| \le \xi, \;\;"
                 r"\left|\tfrac{w_j}{w_W} - a_{jW}\right| \le \xi, \;\; \textstyle\sum_j w_j = 1")
        st.subheader("Modelo Bayesiano hierárquico")
        st.markdown(R.METHOD_TEXT[1])
        st.latex(r"""
\begin{aligned}
A_B^k \mid w^k &\sim \text{Multinomial}\!\left(\tfrac{1/w^k}{\sum_j 1/w_j^k}\right),
\qquad A_W^k \mid w^k \sim \text{Multinomial}(w^k) \\
w^k \mid w^*, \gamma &\sim \text{Dirichlet}(\gamma \, w^*), \qquad k = 1,\dots,K \\
\gamma &\sim \text{Gamma}(0.01,\, 0.01), \qquad w^* \sim \text{Dirichlet}(\mathbf{1})
\end{aligned}""")
        st.markdown(R.METHOD_TEXT[2])
        st.latex(r"P(c_i \succ c_j) = \frac{1}{S}\sum_{s=1}^{S} \mathbb{1}\left[w^{*(s)}_i > w^{*(s)}_j\right]")
        st.subheader("Consistência")
        st.markdown(R.METHOD_TEXT[3])
        st.latex(r"CR = \frac{\xi^*}{CI(a_{BW})}, \qquad CR^I = \max_j \frac{|a_{Bj}\, a_{jW} - a_{BW}|}{a_{BW}^2 - a_{BW}}")
        st.caption("Os limiares de CR^I seguem a tabela de Liang et al. (2020) e os valores de CI seguem Rezaei (2015); "
                   "confirme-os nas publicações originais antes de os citar num trabalho académico.")
        st.subheader("Ordenação de alternativas")
        st.markdown(R.METHOD_TEXT[4])
        st.latex(r"\text{TOPSIS: } C_i = \frac{D_i^-}{D_i^+ + D_i^-} \qquad"
                 r"\text{VIKOR: } Q_i = v\frac{S_i - S^*}{S^- - S^*} + (1-v)\frac{R_i - R^*}{R^- - R^*}")
        st.subheader("Referências")
        for ref in R.REFERENCES:
            st.markdown(f"- {ref}")
    with learn_tabs[4]:
        st.markdown(L.GLOSSARIO)
    with learn_tabs[5]:
        st.caption("Pense na resposta antes de abrir cada pergunta.")
        for q, ans in L.QUIZ:
            with st.expander(q):
                st.markdown(ans)


# ---------------------------------------------------------------------------
# 4. Resultados
# ---------------------------------------------------------------------------


def _input_hash() -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(bo_cur.fillna(-1)).values.tobytes())
    h.update(pd.util.hash_pandas_object(ow_cur.fillna(-1)).values.tobytes())
    h.update(repr((criteria, dms, ss.draws, ss.burn, ss.chains, ss.seed, ss.g_a, ss.g_b)).encode())
    return h.hexdigest()


with tabs[3]:
    errors, _, _ = D.validate(bo_cur, ow_cur) if criteria and dms else (["Sem dados."], [], None)
    cur_hash = _input_hash()
    col_run, col_msg = st.columns([1, 3])
    with col_run:
        run = st.button("Executar análise", type="primary", width="stretch", disabled=bool(errors))
    with col_msg:
        if errors:
            st.error("Há problemas nos dados (ver separador 2). A análise fica disponível quando forem corrigidos.")
        else:
            st.caption(f"{len(criteria)} critérios, {len(dms)} decisores, {ss.chains} cadeias × "
                       f"{ss.draws} amostras. Tempo típico: 5 a 30 segundos.")

    if run:
        bar = st.progress(0.0, text="A estimar o modelo (Hamiltonian Monte Carlo)…")
        try:
            res = bayesian_bwm(bo_cur.values, ow_cur.values, criteria, dms, draws=int(ss.draws),
                               burn_in=int(ss.burn), chains=int(ss.chains), seed=int(ss.seed),
                               gamma_prior=(float(ss.g_a), float(ss.g_b)),
                               progress_callback=lambda f: bar.progress(min(f, 1.0),
                                                                        text=f"A estimar o modelo… {f:.0%}"))
            ss.result = {"res": res, "hash": cur_hash}
            bar.empty()
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"Erro na estimação: {exc}")

    if "result" not in ss:
        st.info("Carregue em **Executar análise** para estimar os pesos. Com o caso exemplo, os dados "
                "já estão prontos.", icon="▶️")
    else:
        res = ss.result["res"]
        if ss.result["hash"] != cur_hash:
            st.warning("Os dados ou parâmetros mudaram desde a última execução. Execute de novo para atualizar.")
        if not (set(res.criteria) <= set(criteria) and set(res.decision_makers) <= set(dms)):
            st.warning("Os critérios ou decisores foram alterados. Execute de novo a análise.")
        else:
            bo_r = bo_cur.loc[res.decision_makers, res.criteria]
            ow_r = ow_cur.loc[res.decision_makers, res.criteria]
            cred = float(ss.cred)
            s = res.summary(cred)
            wt = A.weights_table(res, cred)
            cons = A.consistency_table(bo_r, ow_r)
            diag = res.diagnostics()
            alt_res = None
            if alt_cur is not None and len(alt_cur):
                try:
                    alt_res = A.alternatives_analysis(res, alt_cur, crit_types, float(ss.vikor_v))
                except ValueError as exc:
                    st.warning(f"Alternativas não avaliadas: {exc}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Critério mais importante", wt.iloc[0]["Critério"],
                      help="Critério com maior peso médio agregado.")
            m2.metric("Confiança de ser o 1.º", f"{wt.iloc[0]['P(ser o mais importante)']:.0%}",
                      help="Percentagem de amostras em que este critério tem o maior peso.")
            m3.metric("Concordância γ (mediana)", f"{np.median(res.gamma):.1f}",
                      help="Quanto maior, mais os decisores convergem para o mesmo conjunto de pesos.")
            m4.metric("R-hat máximo", f"{np.nanmax(diag['rhat']):.3f}",
                      help="Abaixo de 1,01 indica que as cadeias convergiram.")

            with st.container(border=True):
                st.markdown("**Interpretação automática**")
                for t in A.interpretation(res, cons, alt_res, ss.alt_method):
                    st.markdown(f"- {t}")

            with st.container(border=True):
                st.markdown("**Leitura para o setor (IA, opcional)**")
                api_key = AI.get_api_key(st.secrets if hasattr(st, "secrets") else None)
                ai_state = ss.get("ai_reading")
                ai_key = (ss.result["hash"], ss.alt_method, (ss.get("sector_label") or ""), ss.dec_context)
                if not api_key:
                    st.caption("Para ativar, defina a variável ANTHROPIC_API_KEY (no Render: Environment; localmente: "
                               "`.streamlit/secrets.toml`). Os cálculos não dependem disto.")
                else:
                    st.caption("O Claude redige uma interpretação adaptada ao setor e ao contexto indicados no topo, "
                               "usando apenas os resultados calculados aqui. Reveja sempre o texto.")
                    if st.button("Gerar leitura para o setor", key="ai_btn"):
                        with st.spinner("A redigir a leitura para o setor…"):
                            try:
                                meta_ai = {**_meta_base(), "criteria_df": crit_df_now, "units": crit_units,
                                           "functions": dm_functions}
                                b_ai = R.build_bundle(res, bo_r, ow_r, meta_ai, alt_cur if alt_res else None,
                                                      crit_types, ss.alt_method, float(ss.vikor_v))
                                ss.ai_reading = {"key": ai_key, "text": AI.sector_reading(b_ai, api_key)}
                                ai_state = ss.ai_reading
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"Não foi possível gerar a leitura: {exc}")
                if ai_state:
                    if ai_state["key"] != ai_key:
                        st.warning("Os resultados ou o contexto mudaram desde que esta leitura foi gerada.")
                    st.markdown(ai_state["text"])
                    st.caption("Texto incluído automaticamente nos relatórios (separador 5).")

            rt = st.tabs(["Pesos", "Ranking credal", "Decisores", "Consistência", "Alternativas", "Diagnóstico"])

            with rt[0]:
                st.caption("Pesos agregados do grupo (w*). A barra laranja é o intervalo de credibilidade: "
                           "a faixa onde o verdadeiro peso está com a probabilidade escolhida.")
                st.plotly_chart(C.weights_bar(res.criteria, s), width="stretch")
                st.dataframe(wt, hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.3f")
                                            for c in wt.columns if c not in ("Posição", "Critério")})
                st.caption("Violinos: forma completa da incerteza de cada peso. Quanto mais estreito, mais certeza.")
                st.plotly_chart(C.posterior_violin(res.criteria, res.w_star), width="stretch")

            with rt[1]:
                st.caption("O ranking credal diz com que confiança um critério é mais importante do que outro. "
                           "Uma confiança de 0,50 significa empate prático; 1,00 significa certeza.")
                P = res.credal_matrix()
                thr = st.slider("Mostrar relações com confiança mínima de", 0.50, 0.99, 0.60, 0.01,
                                help="Setas A → B aparecem quando P(A > B) é igual ou superior a este valor. "
                                     "Relações implícitas (A→B→C) não são repetidas.")
                st.graphviz_chart(C.credal_dot(res.criteria, P, res.w_star.mean(axis=0), thr),
                                  width="stretch")
                st.plotly_chart(C.credal_heatmap(res.criteria, P), width="stretch")
                st.caption("Probabilidade de cada critério ocupar cada posição do ranking.")
                st.plotly_chart(C.rank_prob_heatmap(res.criteria, res.rank_probabilities()),
                                width="stretch")

            with rt[2]:
                st.caption("Pesos estimados para cada decisor. O modelo aproxima os pesos individuais do consenso "
                           "(efeito de encolhimento), sobretudo quando os julgamentos individuais são pouco informativos.")
                st.plotly_chart(C.individual_heatmap(res.criteria, res.decision_makers, res.individual_means(),
                                                     res.w_star.mean(axis=0)), width="stretch")
                st.caption("Distância de Jensen-Shannon ao consenso: 0 = igual ao grupo, 1 = totalmente diferente.")
                st.dataframe(A.dm_distance(res), hide_index=True, width="stretch",
                             column_config={"Distância ao consenso (JS)": st.column_config.ProgressColumn(
                                 min_value=0, max_value=1, format="%.3f")})
                st.plotly_chart(C.gamma_hist(res.gamma), width="stretch")
                with st.expander("Comparação com o BWM clássico (modelo linear, decisor a decisor)"):
                    st.caption("Pesos obtidos separadamente para cada decisor e a sua média aritmética. "
                               "O Bayesian BWM substitui esta média por uma agregação probabilística.")
                    st.dataframe(A.classic_weights_table(bo_r, ow_r).style.format("{:.3f}"),
                                 width="stretch")

            with rt[3]:
                st.caption("Três medidas de coerência dos julgamentos. **CR input-based** (Liang et al., 2020): "
                           "aceitável quando fica abaixo do limiar. **CR clássico** = ξ*/CI(a_BW) (Rezaei, 2015): "
                           "quanto mais perto de 0, melhor; valores acima de ~0,1-0,2 justificam rever os "
                           "julgamentos. **ξ** do modelo linear: perto de 0 indica coerência.")
                st.dataframe(cons, hide_index=True, width="stretch",
                             column_config={"CR input-based": st.column_config.NumberColumn(format="%.3f"),
                                            "Limiar": st.column_config.NumberColumn(format="%.4f"),
                                            "CR clássico": st.column_config.NumberColumn(format="%.3f"),
                                            "ξ* (não linear)": st.column_config.NumberColumn(format="%.4f"),
                                            "ξ (BWM linear)": st.column_config.NumberColumn(format="%.4f")})

            with rt[4]:
                if alt_res is None:
                    st.info("Ative **Avaliar alternativas** no separador 3 para ver esta secção.")
                else:
                    mth = ss.alt_method
                    md = alt_res["methods"][mth]
                    st.caption(f"Método principal: **{mth}**. {RK.METHOD_INFO[mth]} Cada método é aplicado a "
                               "todas as amostras dos pesos, por isso cada alternativa tem um intervalo e uma "
                               "probabilidade de ficar em 1.º lugar.")
                    st.plotly_chart(C.alternatives_bar(alt_res["names"], md["raw"], md["metric"],
                                                       mth == "VIKOR", mth), width="stretch")
                    num_cols = [c for c in md["table"].columns if c not in ("Posição", "Alternativa")]
                    st.dataframe(md["table"], hide_index=True, width="stretch",
                                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in num_cols})
                    st.caption("Probabilidade de cada alternativa ocupar cada posição")
                    st.dataframe(md["rank_prob"].style.format("{:.2f}").background_gradient(cmap="Greens", vmin=0,
                                                                                           vmax=1),
                                 width="stretch")

                    st.markdown("**Comparação entre métodos**")
                    st.caption("Se os três métodos concordam, a recomendação é robusta. Se divergem, a escolha "
                               "depende de quanto se aceita compensar um mau desempenho com outro bom.")
                    st.plotly_chart(C.method_first_prob(alt_res), width="stretch")
                    comp = alt_res["comparison"]
                    st.dataframe(comp, hide_index=True, width="stretch",
                                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                                for c in comp.columns if c.startswith("P(") or c == "Posição média"})
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.caption("Correlação de Spearman entre rankings (1 = iguais)")
                        st.dataframe(RK.spearman_matrix(alt_res).style.format("{:.2f}"), width="stretch")
                    with cc2:
                        vc = alt_res["vikor_check"]
                        st.caption(f"Condições de aceitação do VIKOR (pesos médios, v = {alt_res['v']:.2f})")
                        st.markdown(
                            f"- Vantagem aceitável (DQ = {vc['DQ']:.3f}): "
                            f"{'✅ cumprida' if vc['C1_vantagem'] else '⚠️ não cumprida'}\n"
                            f"- Estabilidade: {'✅ cumprida' if vc['C2_estabilidade'] else '⚠️ não cumprida'}\n\n"
                            f"{vc['mensagem']}")

                    st.markdown("**Análise de sensibilidade**")
                    st.caption("Escolha um critério e veja como a pontuação das alternativas muda quando o seu peso "
                               "varia de 0 a 1 (os restantes mantêm as proporções). As linhas pontilhadas laranja "
                               "marcam os pontos onde a alternativa líder muda.")
                    wm = res.w_star.mean(axis=0)
                    sel_c = st.selectbox("Critério a variar", res.criteria, index=int(np.argmax(wm)),
                                         key="sens_crit")
                    ci = res.criteria.index(sel_c)
                    g, raw_s, sw = RK.sensitivity(mth, alt_res["X"], alt_res["is_benefit"], wm, ci, alt_res["v"])
                    st.plotly_chart(C.sensitivity_chart(g, raw_s, alt_res["names"], sel_c, float(wm[ci]),
                                                        md["metric"], sw), width="stretch")
                    if sw:
                        st.markdown("Mudanças de líder: " + "; ".join(
                            f"com peso ≈ {p:.2f}, «{alt_res['names'][a]}» → «{alt_res['names'][b]}»"
                            for p, a, b in sw))
                    else:
                        st.success(f"A alternativa líder não muda em nenhum valor do peso de «{sel_c}».")
                    with st.expander("Desempenho normalizado (SAW, 0 = pior, 1 = melhor)"):
                        _, N = RK.saw_scores(alt_res["X"], alt_res["is_benefit"], wm[None, :])
                        st.dataframe(pd.DataFrame(N, index=alt_res["names"], columns=res.criteria)
                                     .style.format("{:.2f}"), width="stretch")

            with rt[5]:
                st.caption("Verificações técnicas da estimação. R-hat < 1,01 e ESS > 400 são os valores de referência. "
                           "Se não forem cumpridos, aumente as amostras e o aquecimento na barra lateral.")
                st.dataframe(A.diagnostics_table(res), hide_index=True, width="stretch",
                             column_config={"R-hat": st.column_config.NumberColumn(format="%.3f"),
                                            "ESS": st.column_config.NumberColumn(format="%.0f")})
                st.caption(f"Taxa de aceitação média: {res.acceptance['taxa_aceitacao']:.2f} (alvo ≈ 0,80) · "
                           f"Transições divergentes: {res.acceptance['divergencias']}")
                sel = st.selectbox("Traço das cadeias para o critério", res.criteria)
                st.plotly_chart(C.trace_plot(res.w_star, res.chains, res.draws, res.criteria.index(sel), sel),
                                width="stretch")
                st.caption("Cadeias bem misturadas parecem «ruído» sobreposto, sem tendências nem patamares.")

# ---------------------------------------------------------------------------
# 5. Relatório
# ---------------------------------------------------------------------------

with tabs[4]:
    if "result" not in ss:
        st.info("Execute a análise no separador 4 para gerar o relatório.", icon="📄")
    else:
        res = ss.result["res"]
        if not (set(res.criteria) <= set(criteria) and set(res.decision_makers) <= set(dms)):
            st.warning("Os critérios ou decisores foram alterados. Execute de novo a análise.")
            st.stop()
        st.info("O relatório inclui a síntese interpretada, os dados de entrada, consistência, pesos, ranking "
                "credal, perspetiva de cada decisor, alternativas (se ativas), diagnóstico, metodologia e "
                "referências.", icon="ℹ️")
        offline = st.checkbox("HTML para uso sem Internet (ficheiro maior, ~5 MB)", value=True,
                              help="Desmarque para um HTML leve que carrega os gráficos a partir da Internet.")
        meta = {**_meta_base(), "criteria_df": crit_df_now, "units": crit_units, "functions": dm_functions}
        if ss.get("ai_reading"):
            meta["ai_text"] = ss.ai_reading["text"]
        bo_r = bo_cur.loc[res.decision_makers, res.criteria]
        ow_r = ow_cur.loc[res.decision_makers, res.criteria]
        try:
            bundle = R.build_bundle(res, bo_r, ow_r, meta, alt_cur, crit_types, ss.alt_method, float(ss.vikor_v))
        except ValueError as exc:
            st.warning(f"Alternativas excluídas do relatório: {exc}")
            bundle = R.build_bundle(res, bo_r, ow_r, meta, None, crit_types, ss.alt_method, float(ss.vikor_v))
        slug = "".join(ch if ch.isalnum() else "_" for ch in meta["title"].lower())[:40].strip("_") or "bwm"
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Relatório HTML (interativo)", R.html_report(bundle, offline),
                           f"relatorio_{slug}.html", "text/html", width="stretch", type="primary")
        c2.download_button("Relatório Word", R.docx_report(bundle), f"relatorio_{slug}.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           width="stretch")
        c3.download_button("Tabelas em Excel", R.excel_report(bundle), f"resultados_{slug}.xlsx",
                           width="stretch")
        c4.download_button("Amostras MCMC (CSV)", R.samples_csv(res), f"amostras_{slug}.csv",
                           width="stretch", help="Todas as amostras de w* e γ, para análise externa.")
        st.caption("Dica: abra o HTML no navegador e use Imprimir → Guardar como PDF para obter um PDF.")
        with st.expander("Pré-visualizar síntese"):
            for t in bundle["text"]:
                st.markdown(f"- {t}")
