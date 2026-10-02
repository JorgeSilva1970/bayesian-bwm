"""
Leitura dos resultados adaptada ao setor, redigida pela API do Claude (opcional).

Os números são sempre calculados pela app; o modelo de linguagem apenas redige a interpretação
a partir de um resumo estruturado dos resultados. Sem chave de API, a app funciona normalmente.

Configuração da chave (por ordem de prioridade):
  1. Variável de ambiente ANTHROPIC_API_KEY (Render → Environment)
  2. .streamlit/secrets.toml  →  ANTHROPIC_API_KEY = "..."   (uso local; não enviar para o GitHub)
Modelo: variável ANTHROPIC_MODEL (por omissão, claude-sonnet-5-5).
"""

from __future__ import annotations

import json
import os

import numpy as np

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM = (
    "És um consultor sénior em apoio à decisão multicritério, especialista no Bayesian Best-Worst Method, "
    "TOPSIS e VIKOR. Escreves em português europeu, de forma clara e profissional, para gestores sem formação "
    "estatística avançada. Usa APENAS os números fornecidos; não inventes valores, fontes ou factos sobre "
    "organizações reais. Quando fizeres afirmações sobre o setor, apresenta-as como considerações gerais e "
    "não como factos verificados. Não uses tabelas."
)

INSTRUCOES = """Com base nos resultados abaixo, escreve uma secção «Leitura para o setor» com estes subtítulos
(em Markdown, ### para subtítulos):

### O que os pesos dizem sobre as prioridades
Interpreta os critérios mais e menos importantes à luz do setor e da decisão descrita.

### Consenso e divergência na equipa
Interpreta γ, as distâncias ao consenso e a consistência, ligando-as às funções dos decisores.

### Recomendação sobre as alternativas
(Só se houver alternativas.) Recomenda com base no método principal, na probabilidade de 1.º lugar e na
concordância entre métodos. Se a recomendação não for robusta, di-lo claramente.

### Riscos e próximos passos
3 a 5 pontos concretos e específicos do setor (ex.: o que validar, que dados recolher, que decisores ouvir).

Máximo 450 palavras. Não repitas todos os números; usa os que sustentam cada conclusão."""


def get_api_key(secrets=None) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        if secrets is not None and "ANTHROPIC_API_KEY" in secrets:
            return secrets["ANTHROPIC_API_KEY"]
    except Exception:  # noqa: BLE001  (st.secrets levanta erro se não existir ficheiro)
        return None
    return None


def build_payload(bundle: dict) -> dict:
    """Resumo compacto e estruturado dos resultados (é isto que é enviado à API)."""
    res, meta = bundle["res"], bundle["meta"]
    w = bundle["weights"]
    units = meta.get("units", {})
    crit_df = meta.get("criteria_df")
    descr = {}
    if crit_df is not None:
        descr = {r["Critério"]: {"tipo": r.get("Tipo", ""), "unidade": r.get("Unidade", ""),
                                 "descricao": r.get("Descrição", "")} for _, r in crit_df.iterrows()}
    payload = {
        "decisao": meta.get("title", ""),
        "setor": meta.get("sector") or meta.get("industry", ""),
        "contexto": meta.get("context", ""),
        "criterios": descr,
        "decisores": [{"nome": d, "funcao": meta.get("functions", {}).get(d, "")} for d in res.decision_makers],
        "pesos_grupo": [
            {"criterio": r["Critério"], "peso": round(float(r["Peso médio"]), 3),
             "ic95": [round(float(r.iloc[5]), 3), round(float(r.iloc[6]), 3)],
             "p_mais_importante": round(float(r["P(ser o mais importante)"]), 2)}
            for _, r in w.iterrows()],
        "ranking_credal_consecutivo": [{"a": a, "b": b, "P(a>b)": round(p, 2)} for a, b, p in bundle["chain"]],
        "gamma_mediana": round(float(np.median(res.gamma)), 1),
        "distancia_ao_consenso": [
            {"decisor": r["Decisor"], "js": round(float(r["Distância ao consenso (JS)"]), 3),
             "criterio_mais_valorizado": r["Critério que mais valoriza"]}
            for _, r in bundle["distance"].iterrows()],
        "consistencia": [
            {"decisor": r["Decisor"], "melhor": r["Melhor"], "pior": r["Pior"],
             "avaliacao": r["Avaliação"], "CR_classico": None if np.isnan(r["CR clássico"]) else round(float(r["CR clássico"]), 3)}
            for _, r in bundle["cons"].iterrows()],
        "sintese_automatica": bundle["text"],
    }
    if units:
        payload["unidades"] = units
    ar = bundle.get("alt")
    if ar is not None:
        mth = bundle["method"]
        t = ar["methods"][mth]["table"]
        payload["alternativas"] = {
            "metodo_principal": mth,
            "ranking": [{"alternativa": r["Alternativa"], "valor_medio": round(float(r.iloc[2]), 3),
                         "p_primeiro": round(float(r["P(1.º lugar)"]), 2)} for _, r in t.iterrows()],
            "comparacao_metodos": json.loads(ar["comparison"].round(2).to_json(orient="records", force_ascii=False)),
            "vikor": ar["vikor_check"]["mensagem"],
        }
    return payload


def sector_reading(bundle: dict, api_key: str, model: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    payload = build_payload(bundle)
    msg = client.messages.create(
        model=model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=1500,
        system=SYSTEM,
        messages=[{"role": "user", "content": INSTRUCOES + "\n\nRESULTADOS (JSON):\n"
                   + json.dumps(payload, ensure_ascii=False, indent=1)}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
