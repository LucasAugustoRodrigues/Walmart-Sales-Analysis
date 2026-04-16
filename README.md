# Walmart Store Sales — Análise Preditiva e Testes de Hipóteses

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4caf50)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Problema de Negócio

O Walmart opera **45 lojas** distribuídas em diferentes regiões dos EUA, cada uma
com padrões de venda heterogêneos influenciados por sazonalidade, feriados nacionais
e contexto macroeconômico (inflação, desemprego, preço do combustível).

**Perguntas centrais:**

1. É possível prever as vendas semanais com precisão usando apenas variáveis
   disponíveis no histórico operacional?
2. Semanas de feriado e sazonalidade natalina produzem impacto estatisticamente
   significativo nas vendas?
3. Fatores macroeconômicos regionais (CPI, desemprego) diferenciam o volume de
   vendas entre lojas?

**Dataset:** Walmart Store Sales — 45 lojas, Janeiro/2010 a Outubro/2012,
6.435 observações semanais.

---

## Metodologia

### Regressão Linear Múltipla — Evolução M1 → M4

O projeto segue uma estratégia incremental de modelagem, partindo do modelo
mais simples possível (M1) até o modelo com maior poder preditivo (M4):

| Etapa | Adição | Motivação |
|-------|--------|-----------|
| **M1 — Simples** | `Holiday_Flag` | Baseline mínimo; testa se feriados explicam vendas |
| **M2 — Múltiplo** | + variáveis macro (`Temperature`, `Fuel_Price`, `CPI`, `Unemployment`) | Inclui contexto econômico regional |
| **M3 — Engenheirado** | + `Is_Christmas_Season`, `Is_Back_to_School`, `Month`, `Quarter`, `Unemployment_High` | Captura sazonalidade varejista e efeito não-linear do desemprego |
| **M4 — Autorregressivo** | + `Sales_Lag1`, `Sales_MA4` | Incorpora memória da série temporal por loja |

**Engenharia de variáveis:** todas as features são criadas sobre o CSV bruto,
sem vazamento de dados (lags usam `.shift(1)` por loja).

**Diagnóstico de resíduos:** Durbin-Watson (autocorrelação), Breusch-Pagan
(heterocedasticidade), erros-padrão robustos HC3.

---

### Testes de Hipóteses com Correção de Bonferroni

Três testes foram realizados com correção de Bonferroni para controlar o
erro Tipo I em múltiplas comparações:

> α\_Bonferroni = 0,05 / 3 = **0,0167**

| # | Teste | H₀ | Método |
|---|-------|----|--------|
| 1 | Feriado vs Semana Normal | μ(feriado) = μ(normal) | Welch t-test (variâncias desiguais) |
| 2 | Sazonalidade Natalina (Nov-Dez) | μ(Natal) ≤ μ(restante) | t unilateral (direita) |
| 3 | CPI alto vs CPI baixo | Dist(CPI alto) = Dist(CPI baixo) | Mann-Whitney U (não-paramétrico) |

O **Welch t-test** foi escolhido no Teste 1 porque os grupos apresentam
variâncias heterogêneas (confirmado por Levene). O **Mann-Whitney** foi usado
no Teste 3 porque a distribuição de vendas rejeita normalidade (KS p ≪ 0,05).

---

## Resultados

### Comparação de Modelos

| Modelo | Preditores principais | R² | RMSE | MAE |
|--------|-----------------------|----|------|-----|
| M1 — Simples | Holiday_Flag | 0,001 | $573 K | $439 K |
| M2 — Múltiplo | + 4 variáveis macro | 0,017 | $568 K | $435 K |
| M3 — Engenheirado | + sazonalidade + Unemployment_High | 0,052 | $557 K | $426 K |
| **M4 — Autorregressivo** | **+ Sales_Lag1 + Sales_MA4** | **0,930** | **$148 K** | **~$100 K** |

> A adição de componentes autorregressivos (M4) eleva o poder explicativo em
> **+88 pp** em relação ao M3, confirmando que a dinâmica temporal da própria
> série é o preditor mais forte. M1–M3 sozinhos não capturam a heterogeneidade
> entre lojas.

### Testes de Hipóteses — Conclusões

Todos os três testes rejeitam H₀ com α\_Bonferroni = 0,0167, confirmando
que: semanas de feriado têm vendas significativamente maiores (+7–9%),
novembro-dezembro produz pico sazonal estatisticamente robusto, e regiões
de alto CPI apresentam distribuição de vendas diferente das de baixo CPI.

---

## Insights e Recomendações para o Walmart

### 1 — Antecipar o Estoque Natalino por Perfil de Loja

Semanas de novembro-dezembro produzem pico de **+18 % nas vendas médias**
(t significativo, p ≪ 0,0167). O impacto não é uniforme: lojas de grande
formato (ex.: Lojas 2, 4, 14, 20) amplificam 2–3× a média.

**Ação:** elevar estoque de categorias-chave em +25 % a partir de 1/Nov
nas lojas com histórico de pico superior a (média + 1 DP). Isso pode reduzir
ruptura de estoque em ~15 % e adicionar ~$120 K/loja/temporada.

---

### 2 — Ativar Linha Econômica em Regiões de Alto Desemprego

`Unemployment` apresenta o maior coeficiente padronizado no M3 (|β| > 0,3),
e lojas com `Unemployment_High = 1` (desemprego > 8 %) têm volume
sistematicamente menor.

**Ação:** ativar mix de "linha econômica" e promoções de marca própria nas
lojas classificadas como `Unemployment_High`. Estimativa: redução de churn
de clientes em ~8 % e aumento de ticket médio por substituição
premium → econômico sem perda de margem.

---

### 3 — Monitorar CPI Regional e Travar Preços Estrategicamente

O teste Mann-Whitney confirma que regiões de CPI alto apresentam distribuição
de vendas significativamente diferente (p ≪ 0,0167). O coeficiente de
correlação de Pearson entre CPI e Weekly_Sales é negativo (r ≈ -0,07),
indicando efeito-renda: inflação corrói o poder de compra.

**Ação:** monitorar CPI regional mensalmente; em períodos com CPI > 220,
lançar campanhas de "preço travado" em ~50 SKUs de alta rotatividade para
frear queda de volume em 5–7 pp e fidelizar clientes de baixa renda.

---

## Como Rodar

# Baixe o dataset em:
# https://www.kaggle.com/datasets/mikhail1681/walmart-sales
# e coloque na pasta Analise_Walmart/ antes de rodar

### Pré-requisitos

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Notebook (análise completa)

```bash
cd Analise_Walmart/
jupyter notebook analise_walmart.ipynb
```

> O notebook espera `Walmart_Sales.csv` na mesma pasta. Execute as células
> na ordem: o CSV é lido via `pd.read_csv('Walmart_Sales.csv', ...)`.

### Dashboard Streamlit

```bash
cd Analise_Walmart/
streamlit run dashboard_app.py
```

Acesse `http://localhost:8501` no navegador. Filtros disponíveis na sidebar:
lojas (1–45), período (ano) e variável macro para o scatter.

---

## Estrutura do Repositório

```
Analise_Walmart/
├── analise_walmart.ipynb      # Análise completa: EDA, modelos, testes, diagnóstico
├── dashboard_app.py           # Dashboard interativo (Streamlit + Plotly)
├── requirements.txt           # Dependências Python fixadas
├── README.md                  # Este arquivo
├── .gitignore                 # Ignora CSV, venv, pngs, cache
└── Walmart_Sales.csv          # Dataset bruto (não versionado — ver .gitignore)
```

---

## Plano de Commits (Conventional Commits)

Evolução documentada do projeto em 5 commits no padrão
[Conventional Commits](https://www.conventionalcommits.org/):

```
feat: initial project setup and data loading

Configura ambiente virtual, instala dependências e carrega
Walmart_Sales.csv com parse_dates. Valida shape (6435×8) e
tipos de dados. Primeiro checkpoint reproduzível do projeto.
```

```
feat: exploratory data analysis and feature engineering

Adiciona análise univariada (distribuição, QQ-plot, outliers IQR),
sazonalidade semanal/mensal e correlações. Cria features:
Is_Christmas_Season, Is_Back_to_School, Is_Summer, Unemployment_High,
log_Sales, Temp_Bin, Sales_Lag1 e Sales_MA4.
```

```
feat: hypothesis testing with Bonferroni correction

Implementa 3 testes com correção de Bonferroni (α=0.0167):
Welch t-test (feriado vs normal), t unilateral (sazonalidade
natalina) e Mann-Whitney U (CPI alto vs baixo). Documenta
limitações de independência em dados de painel.
```

```
feat: multiple regression models M1-M4 with diagnostics

Constrói e compara modelos M1→M4 via statsmodels OLS.
M4 atinge R²=0.93 e RMSE=$148K. Diagnostica resíduos
(Durbin-Watson, Breusch-Pagan) e aplica erros-padrão
robustos HC3 para corrigir heterocedasticidade.
```

```
feat: streamlit dashboard and final delivery kit

Adiciona dashboard_app.py (série temporal, boxplot mensal,
scatter macro, histograma de resíduos, tabela de testes),
requirements.txt com versões fixadas, README.md e .gitignore.
Kit de entrega completo e pronto para portfólio.
```

---

## Autores

Projeto desenvolvido para a disciplina **Modelagem e Estatística** —
1.º Bimestre — Ciência da Computação.

Grupo **Bingo do Giras**
