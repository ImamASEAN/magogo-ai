"""
================================================================
  MAGOGO — BSF Farming Optimisation AI  |  Streamlit App
================================================================
  Run with:   streamlit run app.py
================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Import our ML module ─────────────────────────────────────────────────────
from bsf_model import (
    load_and_clean,
    train_models,
    predict_single,
    optimise_conditions,
    FEATURE_LABELS,
)

# ────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MAGOGO — BSF Farming AI",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (used throughout Plotly charts)
# ────────────────────────────────────────────────────────────────────────────
C = {
    "green_dark":  "#1B4332",
    "green_main":  "#2D6A4F",
    "green_mid":   "#52B788",
    "green_light": "#95D5B2",
    "green_pale":  "#D8F3DC",
    "amber":       "#F9A825",
    "amber_light": "#FFF8E1",
    "red":         "#E63946",
    "blue":        "#457B9D",
    "grey":        "#6C757D",
    "bg":          "#F8FAF9",
}

# ────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
.stApp {{ background: {C['bg']}; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {C['green_dark']};
    color: white;
}}
section[data-testid="stSidebar"] * {{
    color: white !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {{
    color: {C['green_light']} !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* ── Metric cards ── */
div[data-testid="metric-container"] {{
    background: white;
    border: 1px solid {C['green_pale']};
    border-left: 4px solid {C['green_mid']};
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
div[data-testid="metric-container"] label {{
    color: {C['grey']} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C['green_dark']} !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}}

/* ── Section headers ── */
.section-header {{
    background: linear-gradient(90deg, {C['green_main']}, {C['green_mid']});
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 16px 0 12px 0;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
}}

/* ── Info boxes ── */
.info-box {{
    background: {C['green_pale']};
    border-left: 4px solid {C['green_mid']};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: {C['green_dark']};
}}
.warn-box {{
    background: {C['amber_light']};
    border-left: 4px solid {C['amber']};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #5D4037;
}}

/* ── Prediction result ── */
.result-card {{
    background: linear-gradient(135deg, {C['green_main']}, {C['green_mid']});
    color: white;
    padding: 24px 32px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(45,106,79,0.25);
    margin: 16px 0;
}}
.result-card .result-number {{
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}}
.result-card .result-label {{
    font-size: 0.9rem;
    opacity: 0.85;
    margin-top: 4px;
}}

/* ── Optimisation card ── */
.opt-row {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid {C['green_pale']};
    font-size: 0.92rem;
}}
.opt-row:last-child {{ border-bottom: none; }}
.opt-label {{ color: {C['grey']}; }}
.opt-value {{ font-weight: 600; color: {C['green_dark']}; }}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
#  DATA & MODEL LOADING  (cached so it only runs once per session)
# ────────────────────────────────────────────────────────────────────────────

DATA_LOCATIONS = [
    "main_dataset.csv",
    "/mnt/user-data/uploads/main_dataset.csv",
]

@st.cache_data(show_spinner=False)
def get_data():
    for path in DATA_LOCATIONS:
        if os.path.exists(path):
            return load_and_clean(path)
    return None

@st.cache_resource(show_spinner=False)
def get_model_bundle():
    df = get_data()
    if df is None:
        return None
    return train_models(df)


# ────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦟 MAGOGO")
    st.markdown("**BSF Farming Optimisation AI**")
    st.markdown("---")
    nav = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔮 Predict Yield", "⚙️ Optimise Conditions"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;opacity:0.7;'>Prototype ML Decision-Support System<br>"
        "Not a certified production tool.<br><br>"
        "Model accuracy improves with more data.</div>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────
#  LOAD DATA & MODEL
# ────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data & training models …"):
    df        = get_data()
    bundle    = get_model_bundle()

if df is None or bundle is None:
    st.error("❌ Could not find `main_dataset.csv`. "
             "Place it in the same folder as `app.py` and restart.")
    st.stop()

harvest_df   = bundle["harvest_df"]
comparison   = bundle["comparison"]
feature_imp  = bundle["feature_imp"]
best_name    = bundle["best_name"]


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if nav == "📊 Dashboard":

    # ── Hero ────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:28px 0 8px 0;'>
      <h1 style='color:#1B4332;margin-bottom:4px;font-size:2rem;'>
        🦟 MAGOGO — BSF Farming Optimisation AI
      </h1>
      <p style='color:#52B788;font-size:1rem;margin:0;'>
        Organic waste → AI optimisation → Black Soldier Fly production insights
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="warn-box">⚠️ <strong>Prototype notice:</strong> '
                'This is an ML decision-support tool trained on '
                f'<strong>{len(harvest_df)} harvest observations</strong>. '
                'Predictions are indicative only. Accuracy improves as more cycles are logged.</div>',
                unsafe_allow_html=True)

    # ── KPI row ─────────────────────────────────────────────
    best_row = comparison[comparison["Model"] == best_name].iloc[0]
    total_harvest = harvest_df["Harvest_Mass"].sum()
    avg_harvest   = harvest_df["Harvest_Mass"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Cycles Logged",   f"{len(harvest_df)}")
    k2.metric("Total BSF Harvested",   f"{total_harvest:.0f} kg")
    k3.metric("Avg Harvest / Cycle",   f"{avg_harvest:.1f} kg")
    k4.metric(f"Best Model MAE",        f"{best_row['mae_mean']:.2f} kg",
              help="Mean Absolute Error via 5-fold CV")
    k5.metric(f"Best Model R²",
              f"{best_row['r2_mean']:.3f}",
              help="R² via 5-fold CV. Negative = model struggles with this dataset size.")

    st.markdown("---")

    # ── Model comparison table ───────────────────────────────
    st.markdown('<div class="section-header">🤖 Model Comparison (5-Fold Cross-Validation)</div>',
                unsafe_allow_html=True)

    fig_cmp = go.Figure()
    colors  = [C["green_main"], C["amber"], C["blue"]]

    for i, row in comparison.iterrows():
        is_best = row["Model"] == best_name
        fig_cmp.add_trace(go.Bar(
            name=row["Model"] + (" ✓ Best" if is_best else ""),
            x=[row["Model"]],
            y=[row["mae_mean"]],
            error_y=dict(type="data", array=[row["mae_std"]], visible=True, color=C["grey"]),
            marker_color=colors[i % len(colors)],
            marker_line_color=C["green_dark"] if is_best else "rgba(0,0,0,0)",
            marker_line_width=3 if is_best else 0,
            text=[f"{row['mae_mean']:.2f} kg"],
            textposition="outside",
        ))

    fig_cmp.update_layout(
        title="Mean Absolute Error by Model (lower = better)",
        yaxis_title="MAE (kg)  ↓ lower is better",
        plot_bgcolor="white", paper_bgcolor="white",
        font_color=C["green_dark"],
        showlegend=False, height=340,
        margin=dict(t=50, b=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # show table too
    disp = comparison.copy()
    disp.columns = ["Model", "MAE Mean", "MAE Std", "R² Mean", "R² Std"]
    for c in disp.columns[1:]:
        disp[c] = disp[c].round(3)
    disp.insert(0, "Best", disp["Model"].apply(lambda x: "✅" if x == best_name else ""))
    st.dataframe(disp.set_index("Model"), use_container_width=True)

    st.markdown("---")

    # ── Time-series & cumulative ─────────────────────────────
    st.markdown('<div class="section-header">📈 BSF Production Over Time</div>',
                unsafe_allow_html=True)

    fig_ts = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Harvest Mass per Cycle", "Cumulative Harvest Mass"),
        vertical_spacing=0.12,
    )

    # Harvest per cycle
    fig_ts.add_trace(go.Scatter(
        x=harvest_df["Date"], y=harvest_df["Harvest_Mass"],
        mode="lines+markers",
        line=dict(color=C["green_main"], width=2.5),
        marker=dict(color=C["green_mid"], size=7, line=dict(color=C["green_dark"], width=1)),
        fill="tozeroy", fillcolor="rgba(82,183,136,0.12)",
        name="Harvest Mass (kg)",
    ), row=1, col=1)

    # Moving average
    ma = harvest_df["Harvest_Mass"].rolling(5, center=True).mean()
    fig_ts.add_trace(go.Scatter(
        x=harvest_df["Date"], y=ma,
        mode="lines",
        line=dict(color=C["amber"], width=2, dash="dot"),
        name="5-cycle moving avg",
    ), row=1, col=1)

    # Cumulative
    fig_ts.add_trace(go.Scatter(
        x=harvest_df["Date"],
        y=harvest_df["Harvest_Mass"].cumsum(),
        mode="lines",
        line=dict(color=C["green_dark"], width=3),
        fill="tozeroy", fillcolor="rgba(27,67,50,0.10)",
        name="Cumulative (kg)",
    ), row=2, col=1)

    fig_ts.update_layout(
        height=480, plot_bgcolor="white", paper_bgcolor="white",
        font_color=C["green_dark"],
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=40, b=10),
    )
    fig_ts.update_yaxes(title_text="kg", gridcolor="#F0F0F0")
    fig_ts.update_xaxes(gridcolor="#F0F0F0")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # ── Environmental trends ─────────────────────────────────
    st.markdown('<div class="section-header">🌡️ Environmental Conditions Over Time</div>',
                unsafe_allow_html=True)

    env_cols = {
        "Temperature":  ("Ambient Temperature (°C)", C["red"]),
        "pH":           ("pH Level",                 C["blue"]),
        "Water_Added":  ("Water Added (L)",           C["green_mid"]),
    }

    # Use all rows (not just harvest rows) for environment
    env_df = df.dropna(subset=["Date"])

    fig_env = make_subplots(
        rows=1, cols=3,
        subplot_titles=[v[0] for v in env_cols.values()],
    )

    for col_idx, (col, (label, colour)) in enumerate(env_cols.items(), start=1):
        sub = env_df.dropna(subset=[col])
        if sub.empty:
            continue
        fig_env.add_trace(go.Scatter(
            x=sub["Date"], y=sub[col],
            mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(size=5),
            name=label,
            showlegend=False,
        ), row=1, col=col_idx)

    fig_env.update_layout(
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        font_color=C["green_dark"],
        margin=dict(t=40, b=10),
    )
    fig_env.update_yaxes(gridcolor="#F0F0F0")
    fig_env.update_xaxes(gridcolor="#F0F0F0")
    st.plotly_chart(fig_env, use_container_width=True)

    st.markdown("---")

    # ── Relationship plots ───────────────────────────────────
    st.markdown('<div class="section-header">🔗 Farming Conditions vs BSF Yield</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_fa = px.scatter(
            harvest_df.dropna(subset=["Feed_Amount", "Harvest_Mass"]),
            x="Feed_Amount", y="Harvest_Mass",
            color="Cycle_Days",
            color_continuous_scale="Greens",
            trendline="ols",
            labels={"Feed_Amount": "Feed Amount (kg)",
                    "Harvest_Mass": "Harvest Mass (kg)",
                    "Cycle_Days": "Cycle Day"},
            title="Feed Amount vs Harvest Mass",
        )
        fig_fa.update_traces(marker=dict(size=9, line=dict(width=1, color=C["green_dark"])))
        fig_fa.update_layout(
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            font_color=C["green_dark"], margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_fa, use_container_width=True)

    with col_b:
        fig_tmp = px.scatter(
            harvest_df.dropna(subset=["Temperature", "Harvest_Mass"]),
            x="Temperature", y="Harvest_Mass",
            color="pH",
            color_continuous_scale="RdYlGn",
            trendline="ols",
            labels={"Temperature": "Ambient Temperature (°C)",
                    "Harvest_Mass": "Harvest Mass (kg)",
                    "pH": "pH"},
            title="Temperature vs Harvest Mass",
        )
        fig_tmp.update_traces(marker=dict(size=9, line=dict(width=1, color=C["green_dark"])))
        fig_tmp.update_layout(
            height=340, plot_bgcolor="white", paper_bgcolor="white",
            font_color=C["green_dark"], margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_tmp, use_container_width=True)

    # ── Feature importance ───────────────────────────────────
    st.markdown('<div class="section-header">🏆 Feature Importance — What Drives BSF Yield?</div>',
                unsafe_allow_html=True)

    fig_fi = go.Figure(go.Bar(
        x=feature_imp["Importance"],
        y=feature_imp["Label"],
        orientation="h",
        marker=dict(
            color=feature_imp["Importance"],
            colorscale=[[0, C["green_light"]], [0.5, C["green_mid"]], [1, C["green_dark"]]],
            line=dict(color=C["green_dark"], width=0.5),
        ),
        text=[f"{v:.3f}" for v in feature_imp["Importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        font_color=C["green_dark"],
        xaxis_title="Importance (Mean Decrease in Impurity)",
        margin=dict(t=20, b=10, l=160),
        xaxis=dict(range=[0, feature_imp["Importance"].max() * 1.25]),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT YIELD
# ════════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Predict Yield":

    st.markdown("""
    <div style='padding:20px 0 8px 0;'>
      <h1 style='color:#1B4332;font-size:1.8rem;'>🔮 Predict BSF Harvest Yield</h1>
      <p style='color:#52B788;'>Enter your farming conditions and get an instant AI yield estimate.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="warn-box">⚠️ Predictions are based on a prototype model trained on '
                f'{len(harvest_df)} cycles. Use as a planning guide, not a guarantee.</div>',
                unsafe_allow_html=True)

    # ── Sliders ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🌱 Input Farming Conditions</div>',
                unsafe_allow_html=True)

    med = bundle["X"].median()  # training medians as sensible defaults

    c1, c2 = st.columns(2)
    with c1:
        feed      = st.slider("🍃 Feed Amount (kg)",       5.0,  25.0, float(round(med.get("Feed_Amount",  12.0), 1)), 0.5)
        cycle     = st.slider("⏱️ Cycle Duration (days)",   1,    350,  int(med.get("Cycle_Days",  200)))
        bulking   = st.slider("🌾 Bulking Agent (kg)",      0.1,   3.0, float(round(med.get("Bulking_Agent", 0.5), 1)), 0.1)
        water     = st.slider("💧 Water Added (L)",         2.0,  20.0, float(round(med.get("Water_Added",   9.0), 1)), 0.5)
    with c2:
        ph        = st.slider("🧪 pH Level",                4.0,  12.0, float(round(med.get("pH",            7.5), 1)), 0.1)
        temp      = st.slider("🌡️ Ambient Temperature (°C)",18.0, 40.0, float(round(med.get("Temperature",   28.0),1)), 0.5)
        reactor_t = st.slider("🔥 Reactor Temperature (°C)",18.0, 45.0, float(round(med.get("Reactor_Temperature", 30.0),1)), 0.5)
        frass     = st.slider("♻️ Frass Mass (kg)",          0.0,  80.0, float(round(med.get("Frass_Mass",   35.0), 0)), 1.0)

    # ── Predict button ───────────────────────────────────────
    if st.button("🚀  Predict Harvest", type="primary", use_container_width=True):
        inputs = {
            "Feed_Amount":         feed,
            "Cycle_Days":          cycle,
            "Bulking_Agent":       bulking,
            "Water_Added":         water,
            "pH":                  ph,
            "Temperature":         temp,
            "Reactor_Temperature": reactor_t,
            "Frass_Mass":          frass,
        }
        prediction = predict_single(bundle, inputs)

        # Result card
        avg = harvest_df["Harvest_Mass"].mean()
        delta_pct = (prediction - avg) / avg * 100
        arrow = "▲" if delta_pct > 0 else "▼"
        colour_d = C["green_mid"] if delta_pct > 0 else C["red"]

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Predicted BSF Harvest Mass</div>
          <div class="result-number">{prediction:.2f} kg</div>
          <div class="result-label" style="margin-top:8px;">
            {arrow} {abs(delta_pct):.1f}% vs historical average ({avg:.1f} kg)
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Radar chart — input conditions vs training averages
        features_show = ["Feed_Amount", "pH", "Temperature", "Water_Added", "Bulking_Agent"]
        labels_show   = ["Feed", "pH", "Temp", "Water", "Bulking"]

        # Normalise 0-1 for radar
        X_tr = bundle["X"]
        mn   = X_tr[features_show].min()
        mx   = X_tr[features_show].max()
        user_norm = [(inputs[f] - mn[f]) / max(mx[f] - mn[f], 1e-6) for f in features_show]
        avg_norm  = [(X_tr[f].mean() - mn[f]) / max(mx[f] - mn[f], 1e-6) for f in features_show]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_norm + [avg_norm[0]],
            theta=labels_show + [labels_show[0]],
            fill="toself", name="Training average",
            line_color=C["grey"], fillcolor="rgba(108,117,125,0.15)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=user_norm + [user_norm[0]],
            theta=labels_show + [labels_show[0]],
            fill="toself", name="Your conditions",
            line_color=C["green_main"], fillcolor="rgba(45,106,79,0.25)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Your Conditions vs Training Average (normalised)",
            height=380, paper_bgcolor="white", font_color=C["green_dark"],
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="info-box">ℹ️ The radar chart shows how your inputs compare to '
                    'the average farming conditions in the training data. Values closer to the edge '
                    'are higher relative to historical data.</div>', unsafe_allow_html=True)

    # ── Actual vs predicted (model fit overview) ─────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📉 Model Fit — Actual vs Predicted</div>',
                unsafe_allow_html=True)

    X_all   = bundle["X"]
    y_all   = bundle["y"]
    y_pred  = bundle["best_model"].predict(X_all)

    fig_avp = go.Figure()
    lo = min(y_all.min(), y_pred.min()) * 0.88
    hi = max(y_all.max(), y_pred.max()) * 1.08

    fig_avp.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", line=dict(color=C["grey"], dash="dash", width=1.5),
        name="Perfect prediction",
    ))
    fig_avp.add_trace(go.Scatter(
        x=y_all, y=y_pred,
        mode="markers",
        marker=dict(color=C["green_mid"], size=9,
                    line=dict(color=C["green_dark"], width=1)),
        name="Observations",
        text=[str(d.date()) for d in harvest_df["Date"]],
        hovertemplate="Actual: %{x:.1f} kg<br>Predicted: %{y:.1f} kg<br>%{text}",
    ))
    fig_avp.update_layout(
        xaxis_title="Actual Harvest Mass (kg)",
        yaxis_title="Predicted Harvest Mass (kg)",
        height=400, plot_bgcolor="white", paper_bgcolor="white",
        font_color=C["green_dark"], margin=dict(t=20),
        xaxis=dict(range=[lo, hi], gridcolor="#F0F0F0"),
        yaxis=dict(range=[lo, hi], gridcolor="#F0F0F0"),
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    best_row = comparison[comparison["Model"] == best_name].iloc[0]
    st.markdown(f'<div class="info-box">Model: <strong>{best_name}</strong> &nbsp;|&nbsp; '
                f'5-Fold CV MAE: <strong>{best_row["mae_mean"]:.2f} ± {best_row["mae_std"]:.2f} kg</strong> &nbsp;|&nbsp; '
                f'R²: <strong>{best_row["r2_mean"]:.3f}</strong></div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — OPTIMISE CONDITIONS
# ════════════════════════════════════════════════════════════════════════════
elif nav == "⚙️ Optimise Conditions":

    st.markdown("""
    <div style='padding:20px 0 8px 0;'>
      <h1 style='color:#1B4332;font-size:1.8rem;'>⚙️ AI Condition Optimisation</h1>
      <p style='color:#52B788;'>
        The AI tests hundreds of farming condition combinations and finds the one
        that is predicted to maximise BSF harvest yield.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="warn-box">⚠️ Optimised conditions are AI suggestions based on the '
                'training data pattern. Always validate in real farming trials.</div>',
                unsafe_allow_html=True)

    # Search space preview
    st.markdown('<div class="section-header">🔍 Search Space</div>', unsafe_allow_html=True)

    sp_col1, sp_col2, sp_col3, sp_col4, sp_col5 = st.columns(5)
    sp_col1.metric("Feed Amount",   "10 – 20 kg",   "step 2.5")
    sp_col2.metric("Temperature",   "25 – 35 °C",   "step 2.5")
    sp_col3.metric("pH",            "6 – 9",        "step 1")
    sp_col4.metric("Water Added",   "5 – 15 L",     "step 2.5")
    sp_col5.metric("Bulking Agent", "0.2 – 2 kg",   "step 0.6")

    st.markdown(
        '<div class="info-box">Other parameters (Cycle Duration, Reactor Temperature, Frass Mass) '
        'are fixed at their median values from the training data during the search.</div>',
        unsafe_allow_html=True,
    )

    if st.button("🔍  Find Optimal Farming Conditions", type="primary", use_container_width=True):
        with st.spinner("Running AI optimisation — testing all combinations …"):
            best = optimise_conditions(bundle)

        pred = best.pop("Predicted_Harvest_Mass")

        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">Maximum Predicted BSF Harvest</div>
          <div class="result-number">{pred:.2f} kg</div>
          <div class="result-label">per cycle under optimal conditions</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">✅ Recommended Conditions</div>',
                    unsafe_allow_html=True)

        UNIT = {
            "Feed_Amount":   "kg",
            "Temperature":   "°C",
            "pH":            "",
            "Water_Added":   "L",
            "Bulking_Agent": "kg",
        }

        # Display as clean rows
        rows_html = ""
        for k, v in best.items():
            label = FEATURE_LABELS.get(k, k)
            unit  = UNIT.get(k, "")
            rows_html += (
                f'<div class="opt-row">'
                f'<span class="opt-label">{label}</span>'
                f'<span class="opt-value">{v:.1f} {unit}</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:white;border:1px solid #D8F3DC;'
            f'border-radius:10px;padding:16px 20px;margin-top:8px;">'
            f'{rows_html}</div>',
            unsafe_allow_html=True,
        )

        # Sensitivity analysis — how yield changes with each key parameter
        st.markdown('<div class="section-header">📊 Sensitivity Analysis</div>',
                    unsafe_allow_html=True)
        st.markdown("How much does predicted yield change as we vary each parameter "
                    "(all others fixed at optimum)?")

        sens_params = {
            "Feed_Amount":   np.linspace(10, 20, 20),
            "Temperature":   np.linspace(25, 35, 20),
            "pH":            np.linspace(6,   9, 20),
            "Water_Added":   np.linspace(5,  15, 20),
        }

        fig_sens = make_subplots(
            rows=2, cols=2,
            subplot_titles=[FEATURE_LABELS.get(k, k) for k in sens_params],
        )
        colours_sens = [C["green_main"], C["red"], C["blue"], C["amber"]]
        pos = [(1,1),(1,2),(2,1),(2,2)]

        for idx, (param, values) in enumerate(sens_params.items()):
            preds = []
            for v in values:
                cond = dict(best)           # start from optimal
                cond[param] = v
                preds.append(predict_single(bundle, cond))

            r, c = pos[idx]
            fig_sens.add_trace(go.Scatter(
                x=values, y=preds,
                mode="lines+markers",
                line=dict(color=colours_sens[idx], width=2.5),
                marker=dict(size=5),
                showlegend=False,
            ), row=r, col=c)

        fig_sens.update_layout(
            height=440, plot_bgcolor="white", paper_bgcolor="white",
            font_color=C["green_dark"], margin=dict(t=40, b=10),
        )
        fig_sens.update_yaxes(title_text="Predicted kg", gridcolor="#F0F0F0")
        fig_sens.update_xaxes(gridcolor="#F0F0F0")
        st.plotly_chart(fig_sens, use_container_width=True)

        st.markdown('<div class="info-box">💡 Flatter lines mean the model is less sensitive '
                    'to that parameter. Steep lines indicate high-leverage variables where '
                    'small changes make a big difference to predicted yield.</div>',
                    unsafe_allow_html=True)
