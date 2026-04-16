"""
dashboard_app.py — Walmart Store Sales · Bingo do Giras
Modelagem e Estatística — 1.º Bimestre

Executar:
    streamlit run dashboard_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e engenharia de variáveis (idêntico ao notebook)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando e processando dados...")
def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê o CSV, aplica a mesma engenharia de variáveis do notebook e
    ajusta o modelo M4. Retorna (df_raw_com_features, df_lag_com_residuos).
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=True)
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # ── Temporais ──────────────────────────────────────────────────────────
    df["Year"]    = df["Date"].dt.year
    df["Month"]   = df["Date"].dt.month
    df["Week"]    = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter

    # ── Sazonais ───────────────────────────────────────────────────────────
    df["Is_Christmas_Season"] = df["Month"].isin([11, 12]).astype(int)
    df["Is_Back_to_School"]   = df["Month"].isin([7, 8]).astype(int)
    df["Is_Summer"]           = df["Month"].isin([6, 7, 8]).astype(int)

    # ── Transformação logarítmica ──────────────────────────────────────────
    df["log_Sales"] = np.log(df["Weekly_Sales"])

    # ── Temperatura categorizada ───────────────────────────────────────────
    df["Temp_Bin"] = pd.cut(
        df["Temperature"],
        bins=[-np.inf, 32, 55, 75, np.inf],
        labels=["Frio", "Fresco", "Ameno", "Quente"],
    )

    # ── Flag econômica ─────────────────────────────────────────────────────
    df["Unemployment_High"] = (df["Unemployment"] > 8.0).astype(int)

    # ── Features autorregressivas ──────────────────────────────────────────
    df["Sales_Lag1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
    df["Sales_MA4"]  = (
        df.groupby("Store")["Weekly_Sales"]
        .transform(lambda x: x.rolling(4, min_periods=1).mean().shift(1))
    )

    # ── Ajuste do modelo M4 ────────────────────────────────────────────────
    df_lag = df.dropna(subset=["Sales_Lag1", "Sales_MA4"]).copy()

    formula_m4 = (
        "Weekly_Sales ~ Holiday_Flag + Temperature + Fuel_Price + "
        "CPI + Unemployment + Is_Christmas_Season + "
        "Is_Back_to_School + Month + C(Quarter) + Unemployment_High + "
        "Sales_Lag1 + Sales_MA4"
    )
    model_m4 = smf.ols(formula_m4, data=df_lag).fit()

    df_lag["M4_Fitted"]    = model_m4.fittedvalues
    df_lag["M4_Residuals"] = model_m4.resid

    return df, df_lag


# ─────────────────────────────────────────────────────────────────────────────
# Carga de dados
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "Walmart_Sales.csv"

if not CSV_PATH.exists():
    st.error(
        f"**Arquivo não encontrado:** `{CSV_PATH}`\n\n"
        "Coloque `Walmart_Sales.csv` na mesma pasta de `dashboard_app.py` e recarregue."
    )
    st.stop()

df_raw, df_m4 = load_and_prepare(str(CSV_PATH))

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — filtros
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")
st.sidebar.markdown("---")

all_stores = sorted(df_m4["Store"].unique().tolist())
selected_stores = st.sidebar.multiselect(
    "Lojas",
    options=all_stores,
    default=all_stores,
    format_func=lambda x: f"Loja {x}",
)

min_year = int(df_m4["Year"].min())
max_year = int(df_m4["Year"].max())
year_range = st.sidebar.slider(
    "Período (ano)",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
    step=1,
)

st.sidebar.markdown("---")
macro_var = st.sidebar.selectbox(
    "Variável macro — Gráfico 3",
    options=["Temperature", "CPI", "Unemployment", "Fuel_Price"],
    format_func=lambda x: {
        "Temperature": "Temperatura (°F)",
        "CPI":         "CPI",
        "Unemployment":"Desemprego (%)",
        "Fuel_Price":  "Preço do Combustível",
    }[x],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Dataset: Walmart Store Sales · 45 lojas · 2010–2012 · 6.435 obs.\n\n"
    "Modelo final: **M4** (Regressão Linear Múltipla + Lag + MA4)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Filtragem
# ─────────────────────────────────────────────────────────────────────────────
if not selected_stores:
    st.warning("Selecione pelo menos uma loja na sidebar.")
    st.stop()

mask = df_m4["Store"].isin(selected_stores) & df_m4["Year"].between(*year_range)
dff  = df_m4[mask].copy()

if dff.empty:
    st.warning("Nenhuma observação para os filtros selecionados.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Cabeçalho e KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛒 Walmart Store Sales — Análise Preditiva")
st.caption(
    "Modelagem e Estatística · 1.º Bimestre · **Bingo do Giras** | "
    f"Exibindo {len(dff):,} observações de {len(selected_stores)} loja(s) "
    f"({year_range[0]}–{year_range[1]})"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Venda Média Semanal",   f"${dff['Weekly_Sales'].mean() / 1e6:.2f}M")
c2.metric("R² — Modelo M4",        "0.93")
c3.metric("RMSE — Modelo M4",      "$148K")
c4.metric("Observações filtradas", f"{len(dff):,}")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1 — Série temporal com faixas de feriado
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1 · Série Temporal de Vendas")
st.caption("Faixas laranja marcam semanas de feriado (`Holiday_Flag = 1`).")

ts_agg = (
    dff.groupby("Date", as_index=False)
    .agg(Sales=("Weekly_Sales", "mean"), Holiday=("Holiday_Flag", "max"))
)

fig_ts = go.Figure()
fig_ts.add_trace(
    go.Scatter(
        x=ts_agg["Date"],
        y=ts_agg["Sales"] / 1e6,
        mode="lines",
        name="Venda Média",
        line=dict(color="royalblue", width=1.5),
    )
)
for hdate in ts_agg.loc[ts_agg["Holiday"] == 1, "Date"]:
    fig_ts.add_vrect(
        x0=hdate - pd.Timedelta(days=3),
        x1=hdate + pd.Timedelta(days=3),
        fillcolor="orange",
        opacity=0.25,
        line_width=0,
    )
fig_ts.update_layout(
    xaxis_title="Data",
    yaxis_title="Weekly Sales (M$)",
    template="plotly_white",
    height=330,
    margin=dict(t=20, b=40),
    legend=dict(orientation="h", y=1.05),
)
st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2 — Boxplot mensal
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("2 · Boxplot Mensal de Vendas")

MONTH_ABBR = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Set", 10: "Out", 11: "Nov", 12: "Dez",
}
dff_box = dff.copy()
dff_box["Mês"] = dff_box["Month"].map(MONTH_ABBR)
month_order = [MONTH_ABBR[m] for m in range(1, 13) if m in dff_box["Month"].unique()]

fig_box = px.box(
    dff_box,
    x="Mês",
    y="Weekly_Sales",
    category_orders={"Mês": month_order},
    labels={"Weekly_Sales": "Weekly Sales (USD)"},
    color_discrete_sequence=["steelblue"],
    template="plotly_white",
    height=370,
)
fig_box.update_layout(margin=dict(t=20, b=40))
st.plotly_chart(fig_box, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gráficos 3 e 4 em colunas
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# ── Gráfico 3 — Scatter macro × vendas ───────────────────────────────────────
with col_left:
    macro_labels = {
        "Temperature": "Temperatura (°F)",
        "CPI":         "CPI",
        "Unemployment":"Desemprego (%)",
        "Fuel_Price":  "Preço do Combustível",
    }
    st.subheader(f"3 · {macro_labels[macro_var]} × Weekly_Sales")

    sample = dff.sample(min(2_000, len(dff)), random_state=42)
    r_val, p_val = stats.pearsonr(sample[macro_var], sample["Weekly_Sales"])
    coef = np.polyfit(sample[macro_var], sample["Weekly_Sales"], 1)
    x_line = np.linspace(sample[macro_var].min(), sample[macro_var].max(), 200)

    fig_sc = go.Figure()
    fig_sc.add_trace(
        go.Scatter(
            x=sample[macro_var],
            y=sample["Weekly_Sales"] / 1e6,
            mode="markers",
            marker=dict(color="steelblue", opacity=0.3, size=4),
            name="Observações",
        )
    )
    fig_sc.add_trace(
        go.Scatter(
            x=x_line,
            y=np.polyval(coef, x_line) / 1e6,
            mode="lines",
            line=dict(color="crimson", width=2),
            name=f"Regressão (r = {r_val:.3f})",
        )
    )
    fig_sc.update_layout(
        xaxis_title=macro_labels[macro_var],
        yaxis_title="Weekly Sales (M$)",
        title=dict(
            text=f"r de Pearson = {r_val:.3f}  |  p = {p_val:.3e}",
            font=dict(size=13),
        ),
        template="plotly_white",
        height=380,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Gráfico 4 — Histograma de resíduos do M4 ─────────────────────────────────
with col_right:
    st.subheader("4 · Distribuição dos Resíduos — Modelo M4")

    fig_res = px.histogram(
        dff,
        x="M4_Residuals",
        nbins=60,
        labels={"M4_Residuals": "Resíduos (USD)"},
        color_discrete_sequence=["coral"],
        template="plotly_white",
        height=380,
    )
    fig_res.add_vline(
        x=0, line_color="black", line_dash="dash", line_width=2,
        annotation_text="μ = 0 (esperado)", annotation_position="top right",
    )
    fig_res.update_layout(
        bargap=0.04,
        margin=dict(t=20, b=40),
        yaxis_title="Frequência",
    )
    st.plotly_chart(fig_res, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabela — Testes de hipóteses (calculados sobre o dataset completo)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("5 · Sumário dos Testes de Hipóteses")
st.caption(
    "Correção de Bonferroni aplicada: α_original = 0,05 / 3 testes → "
    "**α_Bonferroni = 0,0167**. Todos os testes calculados sobre o dataset completo."
)

ALPHA_BONF = 0.05 / 3  # 0.0167

# Teste 1 — Welch t-test (feriado vs normal)
g0 = df_raw.loc[df_raw["Holiday_Flag"] == 0, "Weekly_Sales"]
g1 = df_raw.loc[df_raw["Holiday_Flag"] == 1, "Weekly_Sales"]
t1, p1 = stats.ttest_ind(g1, g0, equal_var=False)

# Teste 2 — t unilateral (Nov-Dez vs restante)
christmas = df_raw.loc[df_raw["Is_Christmas_Season"] == 1, "Weekly_Sales"]
rest      = df_raw.loc[df_raw["Is_Christmas_Season"] == 0, "Weekly_Sales"]
t2, p2_two = stats.ttest_ind(christmas, rest, equal_var=False)
p2 = p2_two / 2  # unilateral direita (t > 0 confirmado no notebook)

# Teste 3 — Mann-Whitney (CPI alto vs baixo)
median_cpi = df_raw["CPI"].median()
high_cpi = df_raw.loc[df_raw["CPI"] >= median_cpi, "Weekly_Sales"]
low_cpi  = df_raw.loc[df_raw["CPI"] <  median_cpi, "Weekly_Sales"]
u3, p3   = stats.mannwhitneyu(high_cpi, low_cpi, alternative="two-sided")

def fmt_conclusion(p: float, alpha: float = ALPHA_BONF) -> str:
    return "✅ Rejeita H₀" if p < alpha else "❌ Não rejeita H₀"

tests_table = pd.DataFrame(
    [
        {
            "Teste":        "1 — Welch t-test",
            "H₀":           "μ(feriado) = μ(normal)",
            "Estatística":  f"t = {t1:.3f}",
            "p-valor":      f"{p1:.4e}",
            "α Bonferroni": f"{ALPHA_BONF:.4f}",
            "Conclusão":    fmt_conclusion(p1),
        },
        {
            "Teste":        "2 — t unilateral (Nov-Dez)",
            "H₀":           "μ(Natal) ≤ μ(restante)",
            "Estatística":  f"t = {t2:.3f}",
            "p-valor":      f"{p2:.4e}",
            "α Bonferroni": f"{ALPHA_BONF:.4f}",
            "Conclusão":    fmt_conclusion(p2),
        },
        {
            "Teste":        "3 — Mann-Whitney U (CPI)",
            "H₀":           "Dist(CPI alto) = Dist(CPI baixo)",
            "Estatística":  f"U = {u3:.0f}",
            "p-valor":      f"{p3:.4e}",
            "α Bonferroni": f"{ALPHA_BONF:.4f}",
            "Conclusão":    fmt_conclusion(p3),
        },
    ]
)

st.dataframe(tests_table, use_container_width=True, hide_index=True)

st.markdown(
    "> **Limitação:** os testes t assumem independência entre observações. "
    "Com dados em painel (mesma loja ao longo do tempo) existe dependência temporal; "
    "os p-valores são indicativos, não definitivos."
)
