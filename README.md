# Bayesian BWM Studio

Aplicação web em Python (Streamlit) para calcular pesos de critérios em decisões de grupo com o
**Bayesian Best-Worst Method** (Mohammadi & Rezaei, 2020), com escolha do setor de atividade,
introdução manual ou por ficheiro, número livre de critérios, decisores e alternativas, caso exemplo
pronto a correr e relatório em HTML, Word e Excel.

## O que a aplicação faz

| Separador | Conteúdo |
|---|---|
| 1. Critérios e decisores | Critérios sugeridos por setor (editáveis), tipo benefício/custo, decisores sem limite |
| 2. Comparações | Vetores Best-to-Others e Others-to-Worst (escala 1-9), deteção automática do melhor/pior, verificação de consistência em tempo real |
| 3. Alternativas | Opcional: ordena alternativas com **TOPSIS, VIKOR e SAW** aplicados a todas as amostras dos pesos (Monte Carlo): probabilidade de 1.º lugar, comparação entre métodos, condições do VIKOR e análise de sensibilidade |
| 4. Resultados | Pesos com intervalos de credibilidade, ranking credal (grafo e matriz), pesos por decisor, concordância γ, consistência, BWM clássico, diagnóstico MCMC, interpretação automática |
| 5. Relatório | HTML interativo, Word, Excel com todas as tabelas, CSV com as amostras MCMC |
| Aprender | Conceito explicado passo a passo, vantagens e limitações, ferramentas grátis, vídeos, modelo matemático, glossário e perguntas de autoavaliação |

Setores incluídos: Indústria transformadora/Automóvel, Saúde, Banca e Seguros, Logística, TI/Software,
Energia, Construção, Retalho, Educação, Turismo, Agroalimentar e Personalizado.

**Casos exemplo (dados fictícios, prontos a correr):**

| Caso | Decisores | Critérios | Alternativas |
|---|---|---|---|
| Automóvel — seleção de fornecedores | Compras, Qualidade, Produção, Sustentabilidade, Finanças | 6 | 4 fornecedores |
| Retalho — localização de nova loja | Expansão, Comercial, Finanças, Operações, Marketing | 6 | 5 localizações |
| Saúde — aquisição de equipamento de ressonância magnética | Direção Clínica, Enfermagem, Radiologia, Administração, Engenharia Biomédica | 6 | 4 equipamentos |
| Didático — 4 critérios e 3 especialistas | 3 especialistas | 4 | 4 fornecedores |

O caso didático reproduz os dados do exemplo Bayesian BWM + TOPSIS da conversa de referência.

## Modelos e formulários para recolha de dados

Na barra lateral, em **📋 Modelos e formulários**, e na pasta `modelos/`:

| Ficheiro | Para quê |
|---|---|
| `modelos/modelo_<setor>.xlsx` | Modelo base por setor, com critérios sugeridos. O `modelo_personalizado.xlsx` serve para qualquer situação. |
| Modelo Excel — configuração atual | Gerado pela app com os critérios, unidades, decisores e valores que estão no ecrã. |
| `modelos/questionario_decisores_base.docx` e botão «Questionário para decisores» | Formulário Word para cada decisor preencher (melhor/pior critério, BO e OW, verificação). |
| Modelo CSV | Só as comparações, para quem prefira CSV. |

O modelo Excel tem 7 folhas: **Instrucoes**, **Projeto** (título, setor/organização, contexto), **Criterios**
(Critério, Tipo, Unidade, Descrição), **Decisores** (Decisor, Função), **Comparacoes** (Decisor, Tipo BO/OW,
uma coluna por critério, Notas), **Alternativas** e **Escala**. Inclui listas pendentes (Benefício/Custo,
BO/OW), validação de valores inteiros de 1 a 9 e destaque automático das células com 1.

Fluxo recomendado: escolher o setor → descarregar o modelo → ajustar critérios → enviar o questionário aos
decisores → transcrever as respostas para **Comparacoes** → carregar o Excel na app.

## Leitura para o setor com IA (opcional)

O botão **Gerar leitura para o setor** (separador Resultados) envia um resumo dos resultados para a API do
Claude, com o setor/organização e o contexto escritos no topo da app, e devolve uma interpretação adaptada,
que passa a constar dos relatórios. Os números são sempre calculados pela app.

- Local: copiar `.streamlit/secrets.toml.example` para `.streamlit/secrets.toml` e colocar a chave
  (este ficheiro não vai para o GitHub).
- Render: Environment → `ANTHROPIC_API_KEY` = a sua chave. Opcional: `ANTHROPIC_MODEL`.
- Sem chave, a app funciona normalmente e o botão não aparece.

## Estrutura

```
bayesian-bwm/
├── app.py                  # interface Streamlit
├── bwm/
│   ├── model.py            # Bayesian BWM (HMC), BWM linear, consistência, diagnósticos
│   ├── data.py             # setores, caso exemplo, leitura de ficheiros, validação
│   ├── analysis.py         # tabelas de resultados e interpretação automática
│   ├── ranking.py          # TOPSIS, VIKOR, SAW com Monte Carlo e sensibilidade
│   ├── learn.py            # conteúdo pedagógico do separador Aprender
│   ├── templates.py        # modelo Excel com validações e questionário Word
│   ├── ai.py               # leitura para o setor via API do Claude (opcional)
│   ├── charts.py           # gráficos Plotly e Matplotlib
│   └── report.py           # relatórios HTML, Word e Excel
├── data/                   # casos exemplo (CSV e Excel)
├── modelos/                # modelos base por setor e questionário
├── tests/test_model.py     # testes automáticos
├── .streamlit/config.toml  # tema e configuração do servidor
├── .vscode/                # configuração de execução e depuração no VS Code
├── .github/workflows/      # testes automáticos no GitHub a cada push
├── render.yaml             # configuração do Render
├── requirements.txt
└── .python-version
```

## 1. Correr localmente no VS Code

```powershell
cd bayesian-bwm
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m streamlit run app.py
```

A app abre em `http://localhost:8501`. No VS Code também pode usar **Run and Debug → Streamlit: app.py**
(configurado em `.vscode/launch.json`), o que permite pôr pontos de paragem no código.

Para correr os testes:

```powershell
python -m pytest -q
```

## 2. Publicar no GitHub

```powershell
git init
git add .
git commit -m "Bayesian BWM Studio: versão inicial"
git branch -M main
git remote add origin https://github.com/JorgeSilva1970/bayesian-bwm.git
git push -u origin main
```

(Crie antes o repositório vazio `bayesian-bwm` em github.com, sem README.) O separador **Actions** do
GitHub passa a correr os testes automaticamente a cada `git push`.

## 3. Publicar no Render

**Opção A — Blueprint (usa o `render.yaml`):** Render → New → Blueprint → escolher o repositório →
Apply.

**Opção B — Web Service manual:** Render → New → Web Service → escolher o repositório e preencher:

| Campo | Valor |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment variable | `PYTHON_VERSION` = `3.12.8` |
| Environment variable (opcional) | `ANTHROPIC_API_KEY` = a sua chave |
| Instance type | Free |

Cada `git push` para `main` volta a publicar a app automaticamente. No plano gratuito, a app adormece
após inatividade e demora cerca de um minuto a acordar no primeiro acesso.

## Formato dos ficheiros

**Comparações (CSV ou folha `Comparacoes` do Excel)** — duas linhas por decisor:

```
Decisor;Tipo;Qualidade;Preço;Prazo de entrega
Ana;BO;1;3;5
Ana;OW;5;2;1
Rui;BO;2;1;4
Rui;OW;3;4;1
```

- `BO` (Best-to-Others): o melhor critério leva 1.
- `OW` (Others-to-Worst): o pior critério leva 1.
- Aceita `;` ou `,` como separador. Pode ter qualquer número de colunas e linhas.

**Excel opcional:** folhas `Criterios` (Critério, Tipo, Descrição) e `Alternativas` (1.ª coluna = nome).
O ficheiro `data/exemplo_completo.xlsx` e os botões de modelo na barra lateral mostram o formato.

## Notas técnicas

- A estimação usa Hamiltonian Monte Carlo implementado em NumPy/SciPy, com gradientes analíticos,
  adaptação do passo e matriz de massa diagonal. Evita PyMC/Stan para manter a instalação leve no Render.
- Os resultados foram validados com testes (gradiente, recuperação de pesos conhecidos, BWM linear,
  consistência e geração dos relatórios) e com R-hat < 1,01 no caso exemplo.
- A consistência é medida de três formas: CR clássico ξ*/CI(a_BW) (Rezaei, 2015), CR input-based
  (Liang et al., 2020) e ξ do modelo linear. Confirme os valores de CI e os limiares nas publicações
  originais antes de os citar num trabalho académico.
- O código da conversa de referência usa um pacote `bayesbwm` que não existe no PyPI. A implementação
  oficial dos autores está em https://github.com/Majeed7/BayesianBWM.

## Referências

- Rezaei, J. (2015). Best-worst multi-criteria decision-making method. *Omega*, 53, 49-57.
- Rezaei, J. (2016). Best-worst multi-criteria decision-making method: Some properties and a linear model. *Omega*, 64, 126-130.
- Mohammadi, M., & Rezaei, J. (2020). Bayesian best-worst method: A probabilistic group decision making model. *Omega*, 96, 102075.
- Liang, F., Brunelli, M., & Rezaei, J. (2020). Consistency issues in the best worst method: Measurements and thresholds. *Omega*, 96, 102175.
- Hwang, C. L., & Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications*. Springer.
- Opricovic, S., & Tzeng, G. H. (2004). Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. *European Journal of Operational Research*, 156(2), 445-455.
