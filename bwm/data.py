"""
Dados de apoio: setores com critérios sugeridos, caso exemplo, modelos de ficheiro,
leitura de CSV/Excel e validação das comparações BWM.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

BENEFICIO, CUSTO = "Benefício", "Custo"

# ---------------------------------------------------------------------------
# Setores e critérios sugeridos (ponto de partida — tudo é editável na app)
# ---------------------------------------------------------------------------

INDUSTRIES: dict[str, dict] = {
    "Indústria transformadora / Automóvel": {
        "decisao": "Seleção de fornecedores estratégicos",
        "criterios": [
            ("Qualidade", BENEFICIO, "Taxa de defeitos, certificações (ISO/IATF), PPAP"),
            ("Preço", CUSTO, "Custo unitário total, incluindo transporte e alfândega"),
            ("Prazo de entrega", CUSTO, "Lead time médio e cumprimento de prazos (OTD)"),
            ("Capacidade tecnológica", BENEFICIO, "I&D, automação, capacidade de co-desenvolvimento"),
            ("Sustentabilidade ambiental", BENEFICIO, "Emissões, ISO 14001, economia circular"),
            ("Estabilidade financeira", BENEFICIO, "Solvabilidade, rating, dependência de clientes"),
        ],
    },
    "Saúde": {
        "decisao": "Aquisição de equipamento médico ou priorização de investimentos hospitalares",
        "criterios": [
            ("Segurança do doente", BENEFICIO, "Redução de riscos e eventos adversos"),
            ("Eficácia clínica", BENEFICIO, "Evidência de resultados em saúde"),
            ("Custo total de posse", CUSTO, "Aquisição, manutenção, consumíveis e formação"),
            ("Facilidade de utilização", BENEFICIO, "Curva de aprendizagem das equipas"),
            ("Interoperabilidade", BENEFICIO, "Integração com sistemas clínicos (HL7/FHIR)"),
            ("Assistência técnica", BENEFICIO, "Tempo de resposta e cobertura do contrato"),
        ],
    },
    "Banca e Seguros": {
        "decisao": "Seleção de parceiro ou solução para operações e canais digitais",
        "criterios": [
            ("Conformidade regulamentar", BENEFICIO, "Aderência a DORA, RGPD, supervisão ASF/BdP"),
            ("Gestão de risco", BENEFICIO, "Controlo de risco operacional e de fraude"),
            ("Custo operacional", CUSTO, "Custo por processo/sinistro/transação"),
            ("Experiência do cliente", BENEFICIO, "NPS, tempo de resposta, omnicanalidade"),
            ("Segurança da informação", BENEFICIO, "Cibersegurança e continuidade de negócio"),
            ("Escalabilidade", BENEFICIO, "Capacidade de acompanhar picos de volume"),
        ],
    },
    "Logística e Transportes": {
        "decisao": "Seleção de operador logístico (3PL)",
        "criterios": [
            ("Custo do serviço", CUSTO, "Tarifas de armazenagem e transporte"),
            ("Fiabilidade de entrega", BENEFICIO, "Entregas no prazo e sem danos"),
            ("Cobertura geográfica", BENEFICIO, "Rede nacional e internacional"),
            ("Capacidade tecnológica", BENEFICIO, "WMS/TMS, rastreabilidade em tempo real"),
            ("Flexibilidade", BENEFICIO, "Resposta a picos e pedidos especiais"),
            ("Pegada carbónica", CUSTO, "Emissões por tonelada-quilómetro"),
        ],
    },
    "Tecnologias de Informação / Software": {
        "decisao": "Seleção de plataforma de software ou fornecedor cloud",
        "criterios": [
            ("Funcionalidades", BENEFICIO, "Cobertura dos requisitos de negócio"),
            ("Custo total de posse", CUSTO, "Licenças, implementação e manutenção"),
            ("Segurança", BENEFICIO, "Certificações, encriptação, gestão de acessos"),
            ("Escalabilidade e desempenho", BENEFICIO, "Resposta sob carga crescente"),
            ("Integração", BENEFICIO, "APIs, conectores, compatibilidade"),
            ("Suporte e comunidade", BENEFICIO, "SLA, documentação, ecossistema"),
        ],
    },
    "Energia": {
        "decisao": "Seleção de projetos de energia renovável",
        "criterios": [
            ("Custo nivelado de energia (LCOE)", CUSTO, "€/MWh ao longo da vida útil"),
            ("Recurso energético", BENEFICIO, "Irradiação, vento, fator de capacidade"),
            ("Impacto ambiental", CUSTO, "Uso do solo, biodiversidade, ruído"),
            ("Ligação à rede", BENEFICIO, "Capacidade e proximidade do ponto de ligação"),
            ("Risco regulatório", CUSTO, "Licenciamento e estabilidade das regras"),
            ("Aceitação social", BENEFICIO, "Apoio das comunidades locais"),
        ],
    },
    "Construção e Imobiliário": {
        "decisao": "Seleção de empreiteiro ou de localização de empreendimento",
        "criterios": [
            ("Preço da proposta", CUSTO, "Valor global da empreitada"),
            ("Prazo de execução", CUSTO, "Duração prevista da obra"),
            ("Experiência e referências", BENEFICIO, "Obras semelhantes concluídas"),
            ("Segurança em obra", BENEFICIO, "Índices de sinistralidade e planos de SST"),
            ("Qualidade técnica", BENEFICIO, "Metodologia, materiais, equipa técnica"),
            ("Capacidade financeira", BENEFICIO, "Garantias e solidez económica"),
        ],
    },
    "Retalho e Distribuição": {
        "decisao": "Seleção de localização para nova loja",
        "criterios": [
            ("Tráfego pedonal", BENEFICIO, "Fluxo diário de potenciais clientes"),
            ("Renda", CUSTO, "Custo mensal do espaço"),
            ("Concorrência", CUSTO, "Densidade de concorrentes diretos"),
            ("Acessibilidade", BENEFICIO, "Transportes públicos e estacionamento"),
            ("Perfil demográfico", BENEFICIO, "Adequação ao público-alvo"),
            ("Visibilidade", BENEFICIO, "Montra, sinalética, exposição"),
        ],
    },
    "Educação": {
        "decisao": "Seleção de plataforma digital de ensino e aprendizagem",
        "criterios": [
            ("Adequação pedagógica", BENEFICIO, "Alinhamento com o currículo e aprendizagens essenciais"),
            ("Facilidade de utilização", BENEFICIO, "Usabilidade para alunos e professores"),
            ("Acessibilidade e inclusão", BENEFICIO, "Suporte a necessidades educativas específicas"),
            ("Custo", CUSTO, "Licenças por aluno/escola"),
            ("Proteção de dados", BENEFICIO, "Conformidade RGPD para menores"),
            ("Formação e apoio", BENEFICIO, "Capacitação docente e suporte técnico"),
        ],
    },
    "Turismo e Hotelaria": {
        "decisao": "Priorização de investimentos numa unidade hoteleira",
        "criterios": [
            ("Retorno do investimento", BENEFICIO, "Impacto esperado no RevPAR e margem"),
            ("Satisfação do hóspede", BENEFICIO, "Avaliações e intenção de regresso"),
            ("Custo de implementação", CUSTO, "Investimento inicial"),
            ("Sustentabilidade", BENEFICIO, "Eficiência energética e certificações"),
            ("Diferenciação", BENEFICIO, "Vantagem face à concorrência local"),
            ("Complexidade operacional", CUSTO, "Impacto nas equipas e processos"),
        ],
    },
    "Agroalimentar": {
        "decisao": "Seleção de fornecedores de matéria-prima",
        "criterios": [
            ("Segurança alimentar", BENEFICIO, "HACCP, certificações, rastreabilidade"),
            ("Preço", CUSTO, "Custo por tonelada"),
            ("Regularidade de abastecimento", BENEFICIO, "Capacidade de fornecer todo o ano"),
            ("Qualidade organolética", BENEFICIO, "Sabor, textura, calibre"),
            ("Práticas sustentáveis", BENEFICIO, "Produção biológica/integrada, água"),
            ("Proximidade", BENEFICIO, "Distância e frescura"),
        ],
    },
    "Personalizado": {
        "decisao": "Problema de decisão definido pelo utilizador",
        "criterios": [
            ("Critério 1", BENEFICIO, ""),
            ("Critério 2", BENEFICIO, ""),
            ("Critério 3", BENEFICIO, ""),
            ("Critério 4", CUSTO, ""),
        ],
    },
}


def industry_criteria_df(industry: str) -> pd.DataFrame:
    rows = INDUSTRIES[industry]["criterios"]
    return pd.DataFrame(rows, columns=["Critério", "Tipo", "Descrição"])


def default_dm_df(k: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"Decisor": [f"Decisor {i + 1}" for i in range(k)],
                         "Função": [""] * k})


# ---------------------------------------------------------------------------
# Casos exemplo (dados ilustrativos, construídos para fins didáticos)
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict] = {
    "Automóvel — seleção de fornecedores": {
        "industry": "Indústria transformadora / Automóvel",
        "title": "Seleção de fornecedores de componentes — fábrica de componentes automóveis",
        "context": (
            "Uma fábrica de componentes automóveis precisa de escolher um fornecedor estratégico de peças "
            "injetadas. Cinco responsáveis de áreas diferentes avaliaram seis critérios com o Best-Worst Method. "
            "O objetivo é obter pesos de grupo que reflitam as várias perspetivas, medir o grau de certeza do "
            "ranking e aplicar os pesos a quatro fornecedores candidatos."
        ),
        "dms": [("Diretora de Compras", "Compras"), ("Engenheiro de Qualidade", "Qualidade"),
                ("Gestor de Produção", "Produção"), ("Responsável de Sustentabilidade", "Sustentabilidade"),
                ("Diretor Financeiro", "Finanças")],
        #        Qual  Preço Prazo Tec  Sust  Fin
        "bo": [[2, 1, 2, 4, 6, 3], [1, 3, 3, 2, 5, 4], [2, 3, 1, 3, 7, 4], [2, 6, 4, 3, 1, 4], [2, 2, 3, 4, 6, 1]],
        "ow": [[3, 6, 3, 2, 1, 2], [5, 2, 2, 3, 1, 2], [4, 3, 7, 2, 1, 2], [4, 1, 2, 2, 6, 2], [3, 4, 2, 2, 1, 6]],
        "alt": {"Fornecedor Alfa (PT)": [92, 14.5, 12, 8, 7, 8],
                "Fornecedor Beta (ES)": [85, 12.8, 9, 6, 5, 7],
                "Fornecedor Gama (PL)": [78, 10.9, 18, 5, 4, 6],
                "Fornecedor Delta (DE)": [95, 16.2, 14, 9, 9, 9]},
        "units": ["pontuação 0-100", "€/unidade", "dias", "escala 1-10", "escala 1-10", "escala 1-10"],
    },
    "Retalho — localização de nova loja": {
        "industry": "Retalho e Distribuição",
        "title": "Localização de uma nova loja de proximidade — cadeia de supermercados",
        "context": (
            "Uma cadeia de supermercados de proximidade vai abrir uma nova loja na área metropolitana e tem "
            "cinco localizações em estudo. As direções de Expansão, Comercial, Financeira, Operações e Marketing "
            "têm prioridades diferentes: tráfego, custo do espaço, adequação ao público-alvo, acessibilidade e "
            "visibilidade da marca. O Bayesian BWM junta estas perspetivas num único conjunto de pesos com "
            "incerteza quantificada."
        ),
        "dms": [("Diretor de Expansão", "Expansão"), ("Diretora Comercial", "Comercial"),
                ("Diretor Financeiro", "Finanças"), ("Gestora de Operações de Loja", "Operações"),
                ("Analista de Marketing", "Marketing")],
        #        Tráf Renda Conc Aces Perf Visib
        "bo": [[1, 2, 3, 2, 2, 6], [2, 7, 3, 2, 1, 2], [2, 1, 3, 4, 3, 8], [2, 3, 7, 1, 2, 2], [2, 8, 3, 3, 2, 1]],
        "ow": [[6, 3, 2, 3, 3, 1], [4, 1, 2, 3, 7, 3], [4, 8, 3, 2, 3, 1], [4, 2, 1, 7, 3, 3], [4, 1, 2, 3, 4, 8]],
        "alt": {"Centro histórico": [9500, 7800, 6, 8, 6, 9],
                "Bairro residencial": [4200, 3200, 2, 6, 9, 5],
                "Centro comercial": [12000, 9500, 8, 9, 7, 7],
                "Interface de transportes": [15000, 8800, 4, 10, 5, 8],
                "Zona empresarial": [6000, 4500, 3, 7, 6, 6]},
        "units": ["pessoas/dia", "€/mês", "concorrentes num raio de 500 m", "escala 1-10",
                  "escala 1-10", "escala 1-10"],
    },
    "Saúde — aquisição de equipamento de ressonância magnética": {
        "industry": "Saúde",
        "title": "Aquisição de equipamento de ressonância magnética — hospital público",
        "context": (
            "Um hospital vai substituir o equipamento de ressonância magnética e recebeu quatro propostas. "
            "A comissão de avaliação junta a Direção Clínica, a Enfermagem, a Radiologia, a Administração e a "
            "Engenharia Biomédica. Cada área valoriza aspetos diferentes: segurança e eficácia clínica, "
            "facilidade de utilização, custo total de posse, integração com os sistemas do hospital e "
            "assistência técnica."
        ),
        "dms": [("Diretor Clínico", "Direção Clínica"), ("Enfermeira Gestora", "Enfermagem"),
                ("Técnico Superior de Radiologia", "Radiologia"), ("Administrador Hospitalar", "Administração"),
                ("Engenheira Biomédica", "Engenharia Biomédica")],
        #        Seg  Efic Custo Usab Inter Assist
        "bo": [[2, 1, 8, 4, 4, 4], [1, 3, 8, 2, 5, 3], [2, 2, 5, 1, 3, 3], [3, 3, 1, 7, 3, 2], [2, 3, 3, 6, 2, 1]],
        "ow": [[5, 8, 1, 2, 2, 2], [8, 3, 1, 4, 2, 3], [3, 3, 1, 5, 2, 2], [3, 3, 7, 1, 2, 3], [3, 2, 2, 1, 4, 6]],
        "alt": {"Equipamento A (1,5 T)": [8, 7, 2.1, 8, 7, 7],
                "Equipamento B (3 T)": [8, 9, 3.4, 6, 8, 8],
                "Equipamento C (1,5 T recondicionado)": [7, 6, 1.4, 7, 5, 5],
                "Equipamento D (3 T gama alta)": [9, 9, 4.2, 7, 9, 9]},
        "units": ["escala 1-10", "escala 1-10", "M€ em 10 anos", "escala 1-10", "escala 1-10", "escala 1-10"],
    },
    "Didático — 4 critérios e 3 especialistas": {
        "industry": "Personalizado",
        "title": "Exemplo didático — escolha de fornecedor com 4 critérios",
        "context": (
            "Exemplo pequeno para aprender o método: três especialistas avaliam custo, qualidade, "
            "sustentabilidade e prazo de entrega, e os pesos são aplicados a quatro fornecedores. "
            "Os dados correspondem ao exemplo da conversa de referência (Bayesian BWM + TOPSIS)."
        ),
        "criteria": [("Custo", CUSTO, "Preço de aquisição (€)"),
                     ("Qualidade", BENEFICIO, "Avaliação de qualidade 1-10"),
                     ("Sustentabilidade", BENEFICIO, "Avaliação ambiental 1-10"),
                     ("Prazo", CUSTO, "Prazo de entrega em dias")],
        "dms": [("Especialista 1", ""), ("Especialista 2", ""), ("Especialista 3", "")],
        "bo": [[3, 1, 4, 7], [2, 1, 8, 4], [1, 2, 5, 6]],
        "ow": [[3, 7, 2, 1], [5, 8, 1, 3], [6, 4, 2, 1]],
        "alt": {"Fornecedor A": [250, 8.5, 7.0, 30], "Fornecedor B": [310, 9.5, 9.0, 45],
                "Fornecedor C": [200, 6.0, 5.0, 20], "Fornecedor D": [270, 8.0, 8.5, 25]},
        "units": ["€", "escala 1-10", "escala 1-10", "dias"],
    },
}

DEFAULT_EXAMPLE = next(iter(EXAMPLES))


def example_case(name: str = DEFAULT_EXAMPLE) -> dict:
    e = EXAMPLES[name]
    if "criteria" in e:
        crit = pd.DataFrame(e["criteria"], columns=["Critério", "Tipo", "Descrição"])
    else:
        crit = industry_criteria_df(e["industry"])
    names = crit["Critério"].tolist()
    dms = pd.DataFrame(e["dms"], columns=["Decisor", "Função"])
    idx = pd.Index(dms["Decisor"], name="Decisor")
    bo_df = pd.DataFrame(e["bo"], columns=names, index=idx)
    ow_df = pd.DataFrame(e["ow"], columns=names, index=idx)
    alt = pd.DataFrame(list(e["alt"].values()), columns=names,
                       index=pd.Index(list(e["alt"].keys()), name="Alternativa"))
    return {"name": name, "industry": e["industry"], "title": e["title"], "context": e["context"],
            "criteria": crit, "dms": dms, "bo": bo_df, "ow": ow_df, "alt": alt,
            "alt_units": dict(zip(names, e["units"]))}


# ---------------------------------------------------------------------------
# Formato de ficheiro (longo): Decisor | Tipo (BO/OW) | critério 1 | critério 2 | ...
# ---------------------------------------------------------------------------


def to_long(bo: pd.DataFrame, ow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dm in bo.index:
        rows.append({"Decisor": dm, "Tipo": "BO", **bo.loc[dm].to_dict()})
        rows.append({"Decisor": dm, "Tipo": "OW", **ow.loc[dm].to_dict()})
    return pd.DataFrame(rows)


def template_long(criteria: list[str], dms: list[str]) -> pd.DataFrame:
    bo = pd.DataFrame(np.nan, index=dms, columns=criteria)
    return to_long(bo, bo.copy())


def template_excel(criteria: list[str], dms: list[str], example: bool | str = False) -> bytes:
    buf = io.BytesIO()
    if example:
        ex = example_case(example if isinstance(example, str) else DEFAULT_EXAMPLE)
        comp = to_long(ex["bo"], ex["ow"])
        crit = ex["criteria"]
        alt = ex["alt"].reset_index()
    else:
        comp = template_long(criteria, dms)
        crit = pd.DataFrame({"Critério": criteria, "Tipo": [BENEFICIO] * len(criteria),
                             "Descrição": [""] * len(criteria)})
        alt = pd.DataFrame(columns=["Alternativa"] + criteria)
    instr = pd.DataFrame({"Instruções": [
        "Folha 'Comparacoes': duas linhas por decisor — Tipo=BO (Best-to-Others) e Tipo=OW (Others-to-Worst).",
        "BO: importância do MELHOR critério face a cada critério (1 = igual, 9 = extremamente mais importante). O melhor critério leva 1.",
        "OW: importância de cada critério face ao PIOR critério (1..9). O pior critério leva 1.",
        "O melhor e o pior critério são detetados automaticamente (valor 1 na linha BO e OW, respetivamente).",
        "Folha 'Criterios' (opcional): Tipo = Benefício (mais é melhor) ou Custo (menos é melhor).",
        "Folha 'Alternativas' (opcional): desempenho de cada alternativa em cada critério, em unidades naturais.",
        "Pode acrescentar colunas (critérios) e linhas (decisores/alternativas) livremente.",
    ]})
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        comp.to_excel(xw, sheet_name="Comparacoes", index=False)
        crit.to_excel(xw, sheet_name="Criterios", index=False)
        alt.to_excel(xw, sheet_name="Alternativas", index=False)
        instr.to_excel(xw, sheet_name="Instrucoes", index=False)
    return buf.getvalue()


def _norm(s: str) -> str:
    return str(s).strip().lower().replace("ç", "c").replace("õ", "o").replace("ã", "a") \
        .replace("é", "e").replace("í", "i").replace("á", "a").replace("ó", "o")


def parse_long(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte a tabela longa em matrizes BO e OW (índice = decisor)."""
    cols = {_norm(c): c for c in df.columns}
    if "decisor" not in cols or "tipo" not in cols:
        raise ValueError("O ficheiro tem de ter as colunas 'Decisor' e 'Tipo' (BO/OW).")
    dcol, tcol = cols["decisor"], cols["tipo"]
    crit_cols = [c for c in df.columns if c not in (dcol, tcol)]
    if len(crit_cols) < 2:
        raise ValueError("São necessárias pelo menos 2 colunas de critérios.")
    df = df.dropna(subset=[dcol]).copy()
    df[tcol] = df[tcol].astype(str).str.strip().str.upper()
    bad = set(df[tcol]) - {"BO", "OW"}
    if bad:
        raise ValueError(f"Valores inválidos na coluna Tipo: {sorted(bad)}. Use apenas BO ou OW.")
    df[dcol] = df[dcol].astype(str).str.strip()
    bo = df[df[tcol] == "BO"].set_index(dcol)[crit_cols]
    ow = df[df[tcol] == "OW"].set_index(dcol)[crit_cols]
    missing = set(bo.index) ^ set(ow.index)
    if missing:
        raise ValueError(f"Decisores sem as duas linhas (BO e OW): {sorted(missing)}")
    if bo.index.duplicated().any() or ow.index.duplicated().any():
        raise ValueError("Há decisores repetidos no ficheiro.")
    ow = ow.loc[bo.index]
    bo.columns = [str(c).strip() for c in bo.columns]
    ow.columns = bo.columns
    return bo.apply(pd.to_numeric, errors="coerce"), ow.apply(pd.to_numeric, errors="coerce")


def read_uploaded(file) -> dict:
    """Lê CSV ou Excel. Excel pode ter as folhas Comparacoes, Criterios e Alternativas."""
    name = file.name.lower()
    out: dict = {}
    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(file, sheet_name=None)
        keymap = {_norm(k): k for k in sheets}
        comp_key = keymap.get("comparacoes") or list(sheets)[0]
        out["bo"], out["ow"] = parse_long(sheets[comp_key])
        if "criterios" in keymap:
            c = sheets[keymap["criterios"]].dropna(how="all")
            if {"Critério", "Tipo"}.issubset(c.columns):
                if "Descrição" not in c.columns:
                    c["Descrição"] = ""
                out["criteria"] = c[["Critério", "Tipo", "Descrição"]].fillna("")
        if "alternativas" in keymap:
            a = sheets[keymap["alternativas"]].dropna(how="all")
            if len(a) and a.shape[1] > 1:
                out["alt"] = a.set_index(a.columns[0])
    else:
        raw = file.getvalue()
        text = raw.decode("utf-8-sig", errors="replace")
        sep = ";" if text.count(";") > text.count(",") else ","
        dec = "," if sep == ";" else "."
        df = pd.read_csv(io.StringIO(text), sep=sep, decimal=dec)
        out["bo"], out["ow"] = parse_long(df)
    return out


def read_alternatives(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        a = pd.read_excel(file)
    else:
        text = file.getvalue().decode("utf-8-sig", errors="replace")
        sep = ";" if text.count(";") > text.count(",") else ","
        a = pd.read_csv(io.StringIO(text), sep=sep, decimal="," if sep == ";" else ".")
    a = a.dropna(how="all")
    return a.set_index(a.columns[0])


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------


def validate(bo: pd.DataFrame, ow: pd.DataFrame) -> tuple[list[str], list[str], pd.DataFrame]:
    """Devolve (erros, avisos, tabela-resumo por decisor)."""
    errors, warnings, rows = [], [], []
    if bo.shape[1] < 2:
        errors.append("Defina pelo menos 2 critérios.")
    if bo.shape[0] < 1:
        errors.append("Defina pelo menos 1 decisor.")
    crit = list(bo.columns)
    for dm in bo.index:
        b = bo.loc[dm].astype(float).values
        w = ow.loc[dm].astype(float).values
        status = "OK"
        if np.isnan(b).any() or np.isnan(w).any():
            errors.append(f"{dm}: existem células por preencher.")
            rows.append({"Decisor": dm, "Melhor": "—", "Pior": "—", "a_BW": np.nan, "Estado": "Incompleto"})
            continue
        if (b < 1).any() or (b > 9).any() or (w < 1).any() or (w > 9).any():
            errors.append(f"{dm}: todos os valores têm de estar entre 1 e 9.")
            status = "Fora da escala"
        best_idx = np.where(b == 1)[0]
        worst_idx = np.where(w == 1)[0]
        if len(best_idx) == 0:
            errors.append(f"{dm}: na linha Best-to-Others o melhor critério tem de ter o valor 1.")
            status = "Sem melhor"
        if len(worst_idx) == 0:
            errors.append(f"{dm}: na linha Others-to-Worst o pior critério tem de ter o valor 1.")
            status = "Sem pior"
        if len(best_idx) and len(worst_idx):
            bi, wi = best_idx[0], worst_idx[0]
            if bi == wi:
                errors.append(f"{dm}: o melhor e o pior critério não podem ser o mesmo.")
                status = "Melhor = pior"
            elif b[wi] != w[bi]:
                warnings.append(
                    f"{dm}: a comparação melhor→pior é {b[wi]:g} na linha BO e {w[bi]:g} na linha OW. "
                    "Deveriam ser iguais (a_BW = a_BW)."
                )
                status = "Aviso"
            if len(best_idx) > 1 or len(worst_idx) > 1:
                warnings.append(f"{dm}: há empates no melhor ou no pior critério (vários valores 1).")
            rows.append({"Decisor": dm, "Melhor": crit[bi], "Pior": crit[wi], "a_BW": b[wi],
                         "Estado": status})
        else:
            rows.append({"Decisor": dm, "Melhor": "—", "Pior": "—", "a_BW": np.nan, "Estado": status})
    return errors, warnings, pd.DataFrame(rows)
