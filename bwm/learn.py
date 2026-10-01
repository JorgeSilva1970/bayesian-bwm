"""Conteúdo pedagógico do separador «Aprender» (texto em Markdown)."""

CONCEITO = """
**Imagine uma comissão que tem de escolher um fornecedor.** Cada pessoa tem prioridades diferentes:
a Compras quer preço, a Qualidade quer zero defeitos, a Produção quer prazos curtos. Antes de comparar
fornecedores, é preciso responder a uma pergunta: *quanto vale cada critério?*

O **Best-Worst Method** (Rezaei, 2015) responde a essa pergunta com um truque simples: em vez de comparar
todos os critérios entre si, cada pessoa escolhe **o melhor** e **o pior** critério e compara apenas esses dois
com os restantes, numa escala de 1 a 9.

| Passo | O que o decisor faz | Exemplo |
|---|---|---|
| 1 | Define os critérios | Qualidade, Preço, Prazo, Tecnologia |
| 2 | Escolhe o **melhor** critério | Qualidade |
| 3 | Escolhe o **pior** critério | Tecnologia |
| 4 | Compara o melhor com todos (**Best-to-Others**) | Qualidade é 3× mais importante do que Preço |
| 5 | Compara todos com o pior (**Others-to-Worst**) | Preço é 4× mais importante do que Tecnologia |

Com *n* critérios bastam **2n − 3** comparações, contra **n(n − 1)/2** no AHP. Com 8 critérios são 13 comparações
em vez de 28, o que torna os julgamentos menos cansativos e mais coerentes.

**E o «Bayesian»?** Quando há vários decisores, o método clássico calcula os pesos de cada um e faz uma média.
A média esconde a discordância e é sensível a opiniões extremas. O **Bayesian BWM** (Mohammadi & Rezaei, 2020)
trata as respostas como dados de um modelo estatístico e estima ao mesmo tempo:

- os **pesos do grupo** (w\\*), com a respetiva incerteza;
- os **pesos de cada decisor**, «puxados» para o consenso na medida em que os dados o justificam;
- o **grau de concordância** (γ) entre os decisores;
- o **ranking credal**: a probabilidade de um critério ser mais importante do que outro, por exemplo
  P(Qualidade > Preço) = 0,94.
"""

AHP_TABLE = {
    "Critérios (n)": [4, 6, 8, 10, 15],
    "Comparações AHP  n(n−1)/2": [6, 15, 28, 45, 105],
    "Comparações BWM  2n−3": [5, 9, 13, 17, 27],
}

VANTAGENS = """
- **Incerteza explícita.** Os pesos vêm com intervalos de credibilidade, não apenas um número.
- **Agregação de grupo com fundamento estatístico.** Substitui a média simples por um modelo hierárquico que
  mede o consenso e a divergência.
- **Menos comparações.** 2n − 3 por decisor, com menor fadiga cognitiva e maior consistência do que o AHP.
- **Ranking credal.** Mostra quando dois critérios estão, na prática, empatados, evitando falsas precisões.
- **Integra-se com métodos de ordenação** (TOPSIS, VIKOR, SAW), propagando a incerteza até à escolha final.
"""

DESVANTAGENS = """
- **Exige cálculo por simulação (MCMC).** Não se resolve numa folha de cálculo simples; precisa de software.
- **Curva de aprendizagem.** Interpretar distribuições a posteriori, R-hat e intervalos de credibilidade
  requer mais formação estatística do que o BWM clássico.
- **Encolhimento para o consenso.** Um decisor muito diferente do grupo aparece «suavizado»; é uma
  propriedade do modelo, mas deve ser explicada a quem lê os resultados.
- **Escala 1-9 continua subjetiva.** O modelo trata a incerteza, mas não corrige julgamentos mal fundamentados.
- **Resultados variam ligeiramente entre execuções** (é uma simulação); fixar a semente torna-os reprodutíveis.
"""

FERRAMENTAS = [
    ("Esta aplicação (Bayesian BWM Studio)", "Grátis", "Python/Streamlit",
     "Pesos, ranking credal, TOPSIS/VIKOR/SAW, relatórios. Corre localmente ou no Render."),
    ("Repositório oficial dos autores — github.com/Majeed7/BayesianBWM", "Grátis (código aberto)",
     "Python e MATLAB + JAGS", "Implementação de referência de Mohammadi & Rezaei. Útil para validar resultados."),
    ("bestworstmethod.com", "Grátis", "Web / Excel",
     "Site do método, com publicações, recursos e solvers para o BWM clássico."),
    ("MetricGate — calculadora BWM", "Grátis", "Web",
     "Calcula pesos e rácio de consistência do BWM clássico para um decisor. Boa para verificar vetores."),
    ("R com rjags, rstan ou brms", "Grátis", "R",
     "Permite reescrever o modelo hierárquico em BUGS/Stan para análises personalizadas."),
    ("Stan / PyMC", "Grátis", "Python, R",
     "Linguagens probabilísticas gerais; o modelo cabe em poucas linhas, mas exige instalação mais pesada."),
]

VIDEOS = [
    ("Bayesian Best-Worst Method — tutoriais",
     "https://www.youtube.com/results?search_query=Bayesian+Best+Worst+Method+tutorial"),
    ("Best-Worst Method (BWM) — introdução e solver em Excel",
     "https://www.youtube.com/results?search_query=Best+Worst+Method+BWM+tutorial+excel"),
    ("TOPSIS passo a passo",
     "https://www.youtube.com/results?search_query=TOPSIS+method+step+by+step"),
    ("VIKOR passo a passo",
     "https://www.youtube.com/results?search_query=VIKOR+method+step+by+step"),
    ("MCMC explicado de forma intuitiva",
     "https://www.youtube.com/results?search_query=MCMC+explained+intuitively"),
]

GLOSSARIO = """
| Termo | Significado |
|---|---|
| **Distribuição a posteriori** | O que sabemos sobre os pesos depois de ver os julgamentos dos decisores. |
| **Intervalo de credibilidade 95%** | Faixa onde o peso verdadeiro está com 95% de probabilidade, segundo o modelo. |
| **w\\*** | Pesos agregados do grupo (o consenso). |
| **γ (gama)** | Concentração: quanto maior, mais os decisores concordam entre si. |
| **Ranking credal** | Probabilidade de um critério ser mais importante do que outro. |
| **MCMC** | Método de simulação que gera milhares de conjuntos plausíveis de pesos. |
| **R-hat** | Compara várias cadeias de simulação; abaixo de 1,01 indica que convergiram. |
| **ESS** | Número efetivo de amostras independentes; acima de 400 é confortável. |
| **CR (rácio de consistência)** | Mede a coerência interna dos julgamentos de um decisor. |
| **TOPSIS** | Ordena alternativas pela proximidade à solução ideal. |
| **VIKOR** | Procura a solução de compromisso entre utilidade de grupo e arrependimento máximo. |
"""

NOTA_GEMINI = """
**Nota sobre o material de referência (conversa com a Gemini).** Os conceitos, vantagens e desvantagens estão
corretos e foram incorporados. Há, no entanto, dois pontos a corrigir:

- O código usa um pacote `bayesbwm` com a classe `BayesianBWM`. Este pacote **não existe no PyPI**, por isso
  `pip install bayesbwm` falha e o código não corre. A implementação oficial dos autores está no repositório
  GitHub *Majeed7/BayesianBWM*; esta aplicação usa uma implementação própria do mesmo modelo.
- A tabela de «saída esperada» do TOPSIS não foi calculada: com os dados indicados, o Fornecedor A fica em
  1.º lugar (não o B). Pode confirmar carregando o caso **Didático** na barra lateral.
"""

QUIZ = [
    ("Com 7 critérios, quantas comparações faz cada decisor no BWM?",
     "2 × 7 − 3 = **11** comparações (no AHP seriam 21)."),
    ("Na linha Best-to-Others, que valor leva o melhor critério?",
     "**1**, porque é comparado consigo próprio."),
    ("O ranking credal indica P(A > B) = 0,52. O que concluir?",
     "Os critérios A e B estão **praticamente empatados**; a ordem entre eles não deve pesar na decisão."),
    ("O γ estimado é baixo. O que significa?",
     "Os decisores **divergem bastante** nas prioridades; vale a pena discutir as diferenças antes de decidir."),
    ("TOPSIS e SAW dão líderes diferentes. Porquê?",
     "O SAW é totalmente compensatório; o TOPSIS penaliza alternativas afastadas do ideal em algum critério. "
     "Líderes diferentes indicam que a escolha depende de quanto se aceita compensar fraquezas."),
]
