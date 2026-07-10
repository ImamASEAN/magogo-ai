"""
MAGOGO — BSF Farming Optimisation AI | Streamlit App
Run: streamlit run app.py
"""

import os, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from bsf_model import (
    load_and_clean, train_models, predict_single,
    optimise_conditions, generate_guidance, yield_interpretation,
    FEATURE_LABELS, GUIDANCE_THRESHOLDS, WHY_OPTIMAL,
)

st.set_page_config(page_title="MAGOGO — BSF Farming AI",
                   page_icon="🦟", layout="wide",
                   initial_sidebar_state="expanded")

# ── ALL text is dark — no white-on-white ─────────────────────────────────────
st.markdown("""
<style>
/* ── Reset to dark text everywhere ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #1a2e1a !important;
    background: #f0f7f2;
}

/* ── Sidebar: dark green with cream text ── */
section[data-testid="stSidebar"] {
    background: #1B4332 !important;
}
section[data-testid="stSidebar"] * {
    color: #d4edda !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #d4edda !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Main area ── */
.stApp { background: #f0f7f2 !important; }
.main .block-container { padding-top: 1.5rem; }

/* ── Metric cards: dark text, green left border ── */
div[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #b7dfc8;
    border-left: 5px solid #2D6A4F;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}
div[data-testid="metric-container"] label {
    color: #4a7c59 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1B4332 !important;
    font-size: 1.65rem !important;
    font-weight: 800 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #2D6A4F !important;
    font-weight: 600 !important;
}

/* ── Section headers ── */
.sec-hdr {
    background: linear-gradient(90deg, #1B4332, #2D6A4F);
    color: #d4edda !important;
    padding: 10px 20px;
    border-radius: 8px;
    margin: 20px 0 12px 0;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.02em;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    color: #1B4332 !important;
    font-weight: 600 !important;
}

/* ── Dataframe text ── */
.dataframe { color: #1a2e1a !important; }

/* ── Slider labels ── */
.stSlider label { color: #1B4332 !important; font-weight: 600 !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #4a7c59 !important; }

/* ── Info / warn boxes ── */
.box-info {
    background: #e8f5e9;
    border-left: 5px solid #2D6A4F;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #1B4332 !important;
    font-size: 0.9rem;
    line-height: 1.6;
}
.box-warn {
    background: #fff8e1;
    border-left: 5px solid #F9A825;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 10px 0;
    color: #5D4037 !important;
    font-size: 0.9rem;
    line-height: 1.6;
}
.box-hero {
    background: linear-gradient(135deg, #1B4332, #2D6A4F);
    color: #d4edda !important;
    padding: 28px 36px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 20px rgba(27,67,50,.3);
    margin: 16px 0;
}
.box-hero .big-num {
    font-size: 3.5rem;
    font-weight: 900;
    color: #ffffff !important;
    letter-spacing: -0.03em;
    line-height: 1;
}
.box-hero .sub {
    font-size: 0.95rem;
    color: #95D5B2 !important;
    margin-top: 6px;
}
.box-hero .msg {
    font-size: 1rem;
    color: #d4edda !important;
    margin-top: 12px;
    font-style: italic;
}

/* ── Tip cards ── */
.tip-card {
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    line-height: 1.6;
}
.tip-card .tip-header {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
}
.tip-card .tip-param {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.tip-card .tip-range {
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.tip-card .tip-body {
    font-size: 0.9rem;
    line-height: 1.55;
}
.tip-high   { background:#fff0f0; border-left:5px solid #E63946; color:#3a0a0a !important; }
.tip-medium { background:#fffbea; border-left:5px solid #F9A825; color:#3a2a00 !important; }
.tip-ok     { background:#e8f5e9; border-left:5px solid #2D6A4F; color:#1B4332 !important; }
.tip-high   .tip-header { color: #c62828 !important; }
.tip-medium .tip-header { color: #e65100 !important; }
.tip-ok     .tip-header { color: #1B4332 !important; }

/* ── Opt rows ── */
.opt-block {
    background: #ffffff;
    border: 1px solid #b7dfc8;
    border-radius: 12px;
    padding: 8px 20px;
    margin: 6px 0;
}
.opt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #e8f5e9;
    font-size: 0.93rem;
}
.opt-row:last-child { border-bottom: none; }
.opt-lbl { color: #4a7c59; font-weight: 500; }
.opt-val { color: #1B4332; font-weight: 700; font-size: 1rem; }
.opt-why { color: #555; font-size: 0.82rem; margin-top: 2px; }

/* ── Steps ── */
.step-box {
    background: #ffffff;
    border: 1px solid #b7dfc8;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
}
.step-num {
    display: inline-block;
    background: #2D6A4F;
    color: #ffffff !important;
    font-weight: 800;
    font-size: 1rem;
    width: 32px; height: 32px;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    margin-right: 10px;
}
.step-title { color: #1B4332 !important; font-weight: 700; font-size: 1rem; }
.step-body  { color: #333 !important; font-size: 0.9rem; margin-top: 6px; line-height: 1.5; }

/* ── Page title ── */
.page-title { color: #1B4332 !important; font-size: 1.9rem; font-weight: 800; margin-bottom: 4px; }
.page-sub   { color: #2D6A4F !important; font-size: 1rem; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

# ── Data & model (cached) ─────────────────────────────────────────────────────
DATA_PATHS = ["main_dataset.csv", "/mnt/user-data/uploads/main_dataset.csv"]

@st.cache_data(show_spinner=False)
def get_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return load_and_clean(p)
    return None

@st.cache_resource(show_spinner=False)
def get_bundle():
    d = get_data()
    return train_models(d) if d is not None else None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 MAGOGO")
    st.markdown("**BSF Farming Optimisation AI**")
    st.markdown("---")
    nav = st.radio("Navigation", [
        "📊 Dashboard",
        "🔮 Predict Yield & Get Tips",
        "⚙️ Find Optimal Conditions",
        "❓ How to Use This App",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:.8rem;color:#95D5B2;line-height:1.7;'>
    <strong style='color:#d4edda;'>What this app does:</strong><br>
    • Analyses your BSF farming data<br>
    • Predicts harvest yield<br>
    • Tells you what to change<br>
    • Finds the best conditions<br><br>
    <em>Prototype — not a certified production tool.</em>
    </div>""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading data and training AI models — please wait …"):
    df     = get_data()
    bundle = get_bundle()

if df is None or bundle is None:
    st.error("❌ Cannot find `main_dataset.csv`. Place it in the same folder as `app.py`.")
    st.stop()

hdf       = bundle["harvest_df"]
comp      = bundle["comparison"]
fi        = bundle["feature_imp"]
best_name = bundle["best_name"]
best_row  = comp[comp["Model"] == best_name].iloc[0]


# ══════════════════════════════════════════════════════════════════════════════
#  HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════
if nav == "❓ How to Use This App":
    st.markdown('<p class="page-title">❓ How to Use MAGOGO AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">A quick guide to getting the most out of this tool</p>', unsafe_allow_html=True)
    st.markdown("---")

    steps = [
        ("Start with the Dashboard",
         "Go to 📊 Dashboard to see an overview of your BSF farming history — production trends, environmental conditions, and which factors affect yield most."),
        ("Predict your next cycle yield",
         "Go to 🔮 Predict Yield & Get Tips. Set the sliders to match your planned farming conditions, then click Predict. You'll get a yield estimate AND a list of specific things to change."),
        ("Follow the guidance tips",
         "Red tips = fix these first (biggest impact). Yellow tips = worth adjusting. Green = you're already in the optimal range. Each tip explains WHY and tells you exactly what to do."),
        ("Find the AI-recommended setup",
         "Go to ⚙️ Find Optimal Conditions and click the button. The AI tests 3,600+ condition combinations in seconds and shows you the setup with the highest predicted yield, with explanations for each recommendation."),
        ("Check sensitivity charts",
         "After optimisation, scroll down to the sensitivity charts. These show how yield changes as you vary each parameter — flat lines mean that parameter doesn't matter much; steep lines mean it's critical."),
    ]

    for i, (title, body) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-box">
            <span class="step-num">{i}</span>
            <span class="step-title">{title}</span>
            <div class="step-body" style="margin-left:42px;">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-hdr">📖 Understanding the Numbers</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="box-info">
        <strong>What is MAE?</strong><br>
        Mean Absolute Error — on average, how many kg the model's prediction is off by.
        A MAE of 2.3 kg means predictions are typically within 2.3 kg of the real harvest.
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="box-info">
        <strong>What is R²?</strong><br>
        A score from –∞ to 1. A score of 1 = perfect predictions.
        A score near 0 or negative means the dataset is too small for the model to find strong patterns yet — this improves as you log more cycles.
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="box-info">
        <strong>Why is the dataset small?</strong><br>
        Only 46 harvest cycles have been recorded. ML models generally need 200+ samples for high accuracy.
        Until then, use predictions as directional guidance, not exact forecasts.
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="box-info">
        <strong>What do the tip colours mean?</strong><br>
        🔴 Red = high priority, biggest impact on yield.<br>
        🟡 Yellow = medium priority, worth adjusting.<br>
        🟢 Green = already in the optimal range — no action needed.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="box-warn">
    ⚠️ <strong>Important:</strong> This is a prototype decision-support system, not a certified agricultural tool.
    Always validate AI recommendations in real farming trials before making large-scale changes.
    Prediction accuracy will improve significantly as more harvest cycles are logged.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Dashboard":
    st.markdown('<p class="page-title">📊 BSF Production Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Overview of your farming history and AI model performance</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box-warn">
    ⚠️ <strong>Prototype notice:</strong> Trained on <strong>{n} harvest cycles</strong>.
    Predictions are directional guidance only — accuracy improves with more data.
    See ❓ How to Use for explanations of all metrics.
    </div>""".format(n=len(hdf)), unsafe_allow_html=True)

    st.markdown("---")

    # KPIs
    st.markdown('<div class="sec-hdr">📌 Key Production Metrics</div>', unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Cycles Logged",     f"{len(hdf)}")
    k2.metric("Total Harvested",   f"{hdf['Harvest_Mass'].sum():.0f} kg")
    k3.metric("Avg per Cycle",     f"{hdf['Harvest_Mass'].mean():.1f} kg")
    k4.metric("Best Cycle",        f"{hdf['Harvest_Mass'].max():.1f} kg")
    k5.metric("AI Model MAE",      f"±{best_row['mae_mean']:.2f} kg",
              help="Average prediction error (lower = better)")

    st.markdown("---")

    # Model comparison
    st.markdown('<div class="sec-hdr">🤖 AI Model Comparison — Which model works best on your data?</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="box-info">
    Three models were tested using 5-fold cross-validation (the data is split 5 different ways and averaged,
    giving a more reliable score than a single test). The model with the <strong>lowest MAE</strong> is
    automatically selected. Error bars show the variation across the 5 splits.
    </div>""", unsafe_allow_html=True)

    fig_cmp = go.Figure()
    clrs = ["#2D6A4F","#F9A825","#457B9D"]
    for i, row in comp.iterrows():
        is_best = row["Model"] == best_name
        fig_cmp.add_trace(go.Bar(
            name=row["Model"] + (" ✓ Selected" if is_best else ""),
            x=[row["Model"]], y=[row["mae_mean"]],
            error_y=dict(type="data",array=[row["mae_std"]],visible=True,color="#888"),
            marker_color=clrs[i % len(clrs)],
            marker_line_color="#1B4332" if is_best else "rgba(0,0,0,0)",
            marker_line_width=3 if is_best else 0,
            text=[f"{row['mae_mean']:.2f} kg"], textposition="outside",
            textfont=dict(color="#1B4332", size=13, family="Inter"),
        ))
    fig_cmp.update_layout(
        yaxis_title="Mean Absolute Error — kg (lower = better)",
        showlegend=False, height=320,
        plot_bgcolor="#ffffff", paper_bgcolor="#f0f7f2",
        font=dict(color="#1B4332", family="Inter"),
        margin=dict(t=20,b=20),
        yaxis=dict(gridcolor="#e0ede6"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # R2 table
    disp = comp.copy()
    disp.columns = ["Model","MAE Mean (kg)","MAE Std","R² Mean","R² Std"]
    for c in disp.columns[1:]: disp[c] = disp[c].round(3)
    disp.insert(0,"Selected", disp["Model"].apply(lambda x:"✅ Best" if x==best_name else ""))
    st.dataframe(disp.set_index("Model"), use_container_width=True)

    st.markdown("---")

    # Time series
    st.markdown('<div class="sec-hdr">📈 Harvest Production Over Time</div>', unsafe_allow_html=True)
    fig_ts = make_subplots(rows=2,cols=1,shared_xaxes=True,
        subplot_titles=("Harvest Mass per Cycle (kg)","Cumulative Total Harvest (kg)"),
        vertical_spacing=0.14)
    fig_ts.add_trace(go.Scatter(x=hdf["Date"],y=hdf["Harvest_Mass"],
        mode="lines+markers",name="Harvest (kg)",
        line=dict(color="#2D6A4F",width=2.5),
        marker=dict(color="#52B788",size=8,line=dict(color="#1B4332",width=1.5)),
        fill="tozeroy",fillcolor="rgba(82,183,136,.15)"),row=1,col=1)
    ma = hdf["Harvest_Mass"].rolling(5,center=True).mean()
    fig_ts.add_trace(go.Scatter(x=hdf["Date"],y=ma,mode="lines",
        name="5-cycle moving avg",
        line=dict(color="#F9A825",width=2,dash="dot")),row=1,col=1)
    fig_ts.add_trace(go.Scatter(x=hdf["Date"],y=hdf["Harvest_Mass"].cumsum(),
        mode="lines",name="Cumulative (kg)",
        line=dict(color="#1B4332",width=3),
        fill="tozeroy",fillcolor="rgba(27,67,50,.12)"),row=2,col=1)
    fig_ts.update_layout(height=480,plot_bgcolor="#ffffff",paper_bgcolor="#f0f7f2",
        font=dict(color="#1B4332",family="Inter"),
        legend=dict(orientation="h",y=-0.07,font=dict(color="#1B4332")),
        margin=dict(t=40,b=10))
    fig_ts.update_yaxes(gridcolor="#e0ede6",tickfont=dict(color="#1B4332"))
    fig_ts.update_xaxes(gridcolor="#e0ede6",tickfont=dict(color="#1B4332"))
    for ann in fig_ts.layout.annotations:
        ann.font.color = "#1B4332"
    st.plotly_chart(fig_ts,use_container_width=True)

    st.markdown("---")

    # Environmental
    st.markdown('<div class="sec-hdr">🌡️ Environmental Conditions Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="box-info">These charts show how temperature, pH and water inputs changed over your recorded cycles. Understanding these trends helps identify which environmental factors coincide with high or low yield periods.</div>', unsafe_allow_html=True)
    env_df = df.dropna(subset=["Date"])
    fig_env = make_subplots(rows=1,cols=3,
        subplot_titles=["Ambient Temperature (°C)","pH Level","Water Added (L)"])
    for ci,(col,clr) in enumerate([("Temperature","#E63946"),("pH","#457B9D"),("Water_Added","#52B788")],1):
        sub = env_df.dropna(subset=[col])
        if sub.empty: continue
        fig_env.add_trace(go.Scatter(x=sub["Date"],y=sub[col],
            mode="lines+markers",line=dict(color=clr,width=2),
            marker=dict(size=4),showlegend=False),row=1,col=ci)
    fig_env.update_layout(height=300,plot_bgcolor="#ffffff",paper_bgcolor="#f0f7f2",
        font=dict(color="#1B4332",family="Inter"),margin=dict(t=40,b=10))
    fig_env.update_yaxes(gridcolor="#e0ede6",tickfont=dict(color="#1B4332"))
    fig_env.update_xaxes(gridcolor="#e0ede6",tickfont=dict(color="#1B4332"))
    for ann in fig_env.layout.annotations: ann.font.color = "#1B4332"
    st.plotly_chart(fig_env,use_container_width=True)

    st.markdown("---")

    # Feature importance
    st.markdown('<div class="sec-hdr">🏆 What Drives BSF Yield? — Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="box-info">This chart shows which input variables the Random Forest model relies on most when making predictions. A higher importance score means that variable has more influence over predicted harvest mass. Use this to prioritise what to monitor and control most carefully.</div>', unsafe_allow_html=True)

    fi_s = fi.sort_values("Importance",ascending=True)
    colours = ["#95D5B2" if v < fi["Importance"].quantile(0.33)
               else ("#52B788" if v < fi["Importance"].quantile(0.66)
               else "#1B4332") for v in fi_s["Importance"]]
    fig_fi = go.Figure(go.Bar(
        x=fi_s["Importance"],y=fi_s["Label"],orientation="h",
        marker=dict(color=colours,line=dict(color="#1B4332",width=0.5)),
        text=[f"{v:.3f}" for v in fi_s["Importance"]],textposition="outside",
        textfont=dict(color="#1B4332",size=11),
    ))
    fig_fi.update_layout(height=320,plot_bgcolor="#ffffff",paper_bgcolor="#f0f7f2",
        font=dict(color="#1B4332",family="Inter"),
        xaxis_title="Feature Importance Score",
        margin=dict(t=10,b=10,l=180),
        xaxis=dict(range=[0,fi["Importance"].max()*1.28],
                   gridcolor="#e0ede6",tickfont=dict(color="#1B4332")),
        yaxis=dict(tickfont=dict(color="#1B4332")))
    st.plotly_chart(fig_fi,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT YIELD & TIPS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Predict Yield & Get Tips":
    st.markdown('<p class="page-title">🔮 Predict Yield & Get Farming Tips</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Set your farming conditions using the sliders, then click Predict to get a yield estimate and personalised guidance on what to improve.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box-warn">
    ⚠️ Prototype model trained on <strong>{n} cycles</strong>. Treat predictions as directional guidance.
    See ❓ How to Use for a full explanation.
    </div>""".format(n=len(hdf)), unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🌱 Step 1 — Set Your Farming Conditions</div>', unsafe_allow_html=True)
    st.markdown('<div class="box-info">Adjust each slider to match your planned or current farming conditions. Hover over any slider label for more context. When ready, click the Predict button below.</div>', unsafe_allow_html=True)

    med = bundle["X"].median()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🍃 Feed & Substrate**")
        feed    = st.slider("Feed Amount (kg) — total organic waste per batch",
                            5.0, 25.0, float(round(med.get("Feed_Amount",12.5),1)), 0.5,
                            help="Total wet weight of organic waste loaded into the reactor per cycle.")
        bulking = st.slider("Bulking Agent (kg) — wood chips, cardboard, etc.",
                            0.1, 3.0, float(round(med.get("Bulking_Agent",0.5),1)), 0.1,
                            help="Dry structural material mixed with waste to maintain airflow.")
        water   = st.slider("Water Added (L) — to adjust substrate moisture",
                            2.0, 20.0, float(round(med.get("Water_Added",9.0),1)), 0.5,
                            help="Water added to the substrate to reach optimal moisture for larvae.")
        ph      = st.slider("pH Level — of the substrate",
                            4.0, 12.0, float(round(med.get("pH",8.0),1)), 0.1,
                            help="Substrate acidity/alkalinity. BSF larvae prefer slightly alkaline (pH 8–10).")
    with col2:
        st.markdown("**🌡️ Temperature & Timing**")
        temp      = st.slider("Ambient Temperature (°C) — outside the reactor",
                              4.0, 15.0, float(round(med.get("Temperature",7.5),1)), 0.5,
                              help="Air temperature around the reactor. Affects larval metabolism.")
        reactor_t = st.slider("Reactor Temperature (°C) — inside the reactor",
                              18.0, 45.0, float(round(med.get("Reactor_Temperature",27.8),1)), 0.5,
                              help="Internal temperature of the BSF reactor where larvae live.")
        cycle     = st.slider("Cycle Duration (days) — how long since start",
                              50, 350, int(med.get("Cycle_Days",200)),
                              help="Number of days elapsed in the current farming cycle.")
        frass     = st.slider("Frass Mass (kg) — waste output collected so far",
                              0.0, 80.0, float(round(med.get("Frass_Mass",0.5),1)), 0.5,
                              help="Total frass (larval excrement + unconsumed waste) collected. Indicates processing progress.")

    user_inputs = {
        "Feed_Amount":feed,"Cycle_Days":cycle,"Bulking_Agent":bulking,
        "Water_Added":water,"pH":ph,"Temperature":temp,
        "Reactor_Temperature":reactor_t,"Frass_Mass":frass,
    }

    st.markdown("---")
    if st.button("🚀  Predict Harvest Yield + Generate Farming Tips",
                 type="primary", use_container_width=True):

        pred   = predict_single(bundle, user_inputs)
        interp = yield_interpretation(pred, bundle)
        tips   = generate_guidance(user_inputs, bundle)

        # Result banner
        arrow = "▲" if interp["delta_pct"] > 0 else "▼"
        st.markdown(f"""
        <div class="box-hero">
          <div style="font-size:.85rem;color:#95D5B2;margin-bottom:6px;">
            {interp['emoji']} Yield Rating: <strong style="color:#ffffff;">{interp['level'].upper()}</strong>
          </div>
          <div class="big-num">{pred:.2f} kg</div>
          <div class="sub">Predicted BSF Harvest Mass</div>
          <div class="sub">{arrow} {abs(interp['delta_pct']):.1f}% vs historical average ({interp['median']:.1f} kg)</div>
          <div class="msg">{interp['message']}</div>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            delta={"reference":interp["median"],"valueformat":".1f","suffix":" kg",
                   "increasing":{"color":"#2D6A4F"},"decreasing":{"color":"#E63946"}},
            number={"suffix":" kg","font":{"size":40,"color":"#1B4332"}},
            title={"text":"Predicted Yield vs Historical Average","font":{"color":"#1B4332","size":14}},
            gauge={
                "axis":{"range":[20,45],"ticksuffix":" kg",
                        "tickfont":{"color":"#1B4332"},"tickcolor":"#1B4332"},
                "bar":{"color":"#2D6A4F"},
                "steps":[
                    {"range":[20,interp["low_threshold"]],"color":"#FFEBEE"},
                    {"range":[interp["low_threshold"],interp["high_threshold"]],"color":"#E8F5E9"},
                    {"range":[interp["high_threshold"],45],"color":"#C8E6C9"},
                ],
                "threshold":{"line":{"color":"#F9A825","width":4},
                             "thickness":0.8,"value":interp["median"]},
            },
        ))
        fig_g.update_layout(height=280,paper_bgcolor="#f0f7f2",
            font=dict(color="#1B4332",family="Inter"),
            margin=dict(t=40,b=10,l=20,r=20))
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown('<div class="box-info">🟡 Yellow line = historical average yield. Green zone = high-yield range. Red zone = low-yield range.</div>', unsafe_allow_html=True)

        # Tips
        st.markdown('<div class="sec-hdr">🌿 Step 2 — Your Personalised Farming Guidance</div>', unsafe_allow_html=True)

        high_tips   = [t for t in tips if t["priority"]=="high"]
        medium_tips = [t for t in tips if t["priority"]=="medium"]
        ok_tips     = [t for t in tips if t["priority"]=="ok"]

        if not high_tips and not medium_tips:
            st.markdown('<div class="box-info">✅ All parameters are within their optimal ranges — great work! No changes needed for this cycle.</div>', unsafe_allow_html=True)

        if high_tips:
            st.markdown("#### 🔴 High Priority — Fix These First")
            st.markdown('<div class="box-info" style="background:#fff0f0;border-color:#E63946;">These parameters are far outside the optimal range and will have the biggest impact on your yield if corrected.</div>', unsafe_allow_html=True)
            for t in high_tips:
                st.markdown(f"""
                <div class="tip-card tip-high">
                  <div class="tip-header">🔴 HIGH PRIORITY</div>
                  <div class="tip-param">{t['label']}</div>
                  <div class="tip-range">
                    Your value: <strong>{t['current_value']:.1f}{t['unit']}</strong>
                    &nbsp;→&nbsp; Target range: <strong>{t['optimal_min']}–{t['optimal_max']}{t['unit']}</strong>
                  </div>
                  <div class="tip-body">💡 {t['tip']}</div>
                </div>""", unsafe_allow_html=True)

        if medium_tips:
            st.markdown("#### 🟡 Medium Priority — Worth Adjusting")
            for t in medium_tips:
                st.markdown(f"""
                <div class="tip-card tip-medium">
                  <div class="tip-header">🟡 MEDIUM PRIORITY</div>
                  <div class="tip-param">{t['label']}</div>
                  <div class="tip-range">
                    Your value: <strong>{t['current_value']:.1f}{t['unit']}</strong>
                    &nbsp;→&nbsp; Target range: <strong>{t['optimal_min']}–{t['optimal_max']}{t['unit']}</strong>
                  </div>
                  <div class="tip-body">💡 {t['tip']}</div>
                </div>""", unsafe_allow_html=True)

        if ok_tips:
            with st.expander("✅ Parameters already in optimal range (click to expand)"):
                for t in ok_tips:
                    st.markdown(f"""
                    <div class="tip-card tip-ok">
                      <div class="tip-header">✅ OPTIMAL</div>
                      <div class="tip-param">{t['label']}</div>
                      <div class="tip-range">Your value: <strong>{t['current_value']:.1f}{t['unit']}</strong>
                      &nbsp; Target: {t['optimal_min']}–{t['optimal_max']}{t['unit']}</div>
                      <div class="tip-body">{t['tip']}</div>
                    </div>""", unsafe_allow_html=True)

        # What-if simulation
        st.markdown("---")
        st.markdown('<div class="sec-hdr">💡 Step 3 — What If You Applied All Tips?</div>', unsafe_allow_html=True)
        st.markdown('<div class="box-info">This simulation takes your current conditions, applies the midpoint of every optimal range for flagged parameters, and re-runs the prediction. It shows the maximum potential improvement if you follow all the guidance above.</div>', unsafe_allow_html=True)

        improved = dict(user_inputs)
        changed  = []
        for t in tips:
            if t["status"] != "optimal":
                p = t["parameter"]
                target = (t["optimal_min"] + t["optimal_max"]) / 2
                improved[p] = target
                changed.append((t["label"], t["current_value"], t["unit"], target))

        improved_pred = predict_single(bundle, improved)
        gain = improved_pred - pred

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Predicted Yield", f"{pred:.2f} kg")
        c2.metric("Potential Yield (all tips applied)", f"{improved_pred:.2f} kg",
                  delta=f"{gain:+.2f} kg")
        c3.metric("Estimated Gain", f"{gain:+.2f} kg",
                  delta=f"{(gain/pred*100):+.1f}%")

        if changed:
            st.markdown("**Changes applied in this simulation:**")
            rows_html = "".join([
                f'<div class="opt-row"><span class="opt-lbl">{lbl}</span>'
                f'<span class="opt-val">{cur:.1f}{u} → {tgt:.1f}{u}</span></div>'
                for lbl, cur, u, tgt in changed
            ])
            st.markdown(f'<div class="opt-block">{rows_html}</div>', unsafe_allow_html=True)

        # Radar
        st.markdown("---")
        st.markdown('<div class="sec-hdr">📡 Conditions Radar — Your Setup vs Optimal Range</div>', unsafe_allow_html=True)
        st.markdown('<div class="box-info">Each axis represents one farming parameter. A value of 0.5 means you are exactly in the middle of the optimal range. Values below 0 mean too low; above 1 mean too high. Aim to keep all points near 0.5.</div>', unsafe_allow_html=True)

        radar_params = ["Feed_Amount","pH","Temperature","Water_Added","Bulking_Agent","Cycle_Days"]
        radar_labels = [FEATURE_LABELS.get(p,p).split(" (")[0] for p in radar_params]

        def norm(val, lo, hi):
            return max(-0.2, min(1.2, (val - lo) / max(hi - lo, 1e-6)))

        user_n  = [norm(user_inputs[p], GUIDANCE_THRESHOLDS[p]["optimal_min"],
                        GUIDANCE_THRESHOLDS[p]["optimal_max"]) for p in radar_params]
        ideal_n = [0.5] * len(radar_params)

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=ideal_n+[ideal_n[0]], theta=radar_labels+[radar_labels[0]],
            fill="toself", name="Optimal (centre of range)",
            line=dict(color="#52B788",width=2),
            fillcolor="rgba(82,183,136,.2)"))
        fig_r.add_trace(go.Scatterpolar(
            r=user_n+[user_n[0]], theta=radar_labels+[radar_labels[0]],
            fill="toself", name="Your conditions",
            line=dict(color="#1B4332",width=2.5),
            fillcolor="rgba(27,67,50,.25)"))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True,range=[-0.2,1.2],
                                tickvals=[0,.5,1],
                                ticktext=["Below opt.","Optimal","Above opt."],
                                tickfont=dict(color="#1B4332",size=10)),
                angularaxis=dict(tickfont=dict(color="#1B4332",size=11))),
            height=380,paper_bgcolor="#f0f7f2",
            font=dict(color="#1B4332",family="Inter"),
            legend=dict(orientation="h",y=-0.08,font=dict(color="#1B4332")),
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMISE CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "⚙️ Find Optimal Conditions":
    st.markdown('<p class="page-title">⚙️ Find Optimal Farming Conditions</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">The AI tests 3,600+ parameter combinations in seconds and identifies the setup most likely to maximise BSF harvest yield.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="box-warn">
    ⚠️ AI suggestions based on training data patterns. Always validate in real farming trials before scaling up.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🔍 Search Space — What the AI Explores</div>', unsafe_allow_html=True)
    st.markdown('<div class="box-info">The AI tests every combination of the ranges below. Other parameters (Cycle Duration, Reactor Temperature, Frass Mass) are fixed at their median values from your training data during the search, then shown in the results.</div>', unsafe_allow_html=True)

    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("Feed Amount",   "10–20 kg",   "7 steps")
    s2.metric("Temperature",   "5–10 °C",    "6 steps")
    s3.metric("pH",            "6–10",       "5 steps")
    s4.metric("Water Added",   "5–15 L",     "5 steps")
    s5.metric("Bulking Agent", "0.5–2 kg",   "4 steps")

    st.markdown("""
    <div class="box-info">
    <strong>How it works:</strong> 7 × 6 × 5 × 5 × 4 = <strong>4,200 combinations</strong> tested simultaneously
    using batch prediction — completes in under 2 seconds.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔍  Run AI Optimisation Now", type="primary", use_container_width=True):
        with st.spinner("🤖 Testing all combinations — this takes just a moment …"):
            best = optimise_conditions(bundle)

        pred_opt = best.pop("Predicted_Harvest_Mass")
        avg      = bundle["median_yield"]
        gain_pct = (pred_opt - avg) / avg * 100

        st.markdown(f"""
        <div class="box-hero">
          <div style="font-size:.85rem;color:#95D5B2;margin-bottom:6px;">
            🏆 OPTIMAL CONDITIONS FOUND
          </div>
          <div class="big-num">{pred_opt:.2f} kg</div>
          <div class="sub">Maximum Predicted BSF Harvest per Cycle</div>
          <div class="sub">▲ {gain_pct:.1f}% above historical average ({avg:.1f} kg)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">✅ Recommended Conditions — With Explanations</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="box-info">Each recommendation includes the reasoning behind it based on your farming data and BSF biology. Use these as your target conditions for the next cycle.</div>', unsafe_allow_html=True)

        UNIT = {"Feed_Amount":"kg","Temperature":"°C","pH":"","Water_Added":"L",
                "Bulking_Agent":"kg","Cycle_Days":"days",
                "Reactor_Temperature":"°C","Frass_Mass":"kg"}

        rows_html = ""
        for k, v in best.items():
            label = FEATURE_LABELS.get(k, k)
            unit  = UNIT.get(k, "")
            why   = WHY_OPTIMAL.get(k, "")
            lo    = GUIDANCE_THRESHOLDS.get(k, {}).get("optimal_min", "—")
            hi    = GUIDANCE_THRESHOLDS.get(k, {}).get("optimal_max", "—")
            rows_html += f"""
            <div style="padding:14px 0;border-bottom:1px solid #e8f5e9;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;
                              letter-spacing:.06em;color:#4a7c59;margin-bottom:3px;">{label}</div>
                  <div style="font-size:1.05rem;font-weight:800;color:#1B4332;">
                    {v:.1f} {unit}
                  </div>
                  <div style="font-size:.8rem;color:#666;margin-top:2px;">
                    Optimal range: {lo}–{hi} {unit}
                  </div>
                </div>
                <div style="background:#2D6A4F;color:#ffffff;padding:3px 10px;
                            border-radius:20px;font-size:.75rem;font-weight:700;
                            white-space:nowrap;align-self:flex-start;">✓ RECOMMENDED</div>
              </div>
              <div style="font-size:.88rem;color:#333;margin-top:8px;
                          background:#f0f7f2;padding:8px 12px;border-radius:6px;
                          line-height:1.55;">
                💡 {why}
              </div>
            </div>"""

        st.markdown(f'<div style="background:#ffffff;border:1px solid #b7dfc8;border-radius:12px;padding:4px 20px;">{rows_html}</div>', unsafe_allow_html=True)

        # Sensitivity analysis
        st.markdown("---")
        st.markdown('<div class="sec-hdr">📊 Sensitivity Analysis — How Each Parameter Affects Yield</div>', unsafe_allow_html=True)
        st.markdown('<div class="box-info">Each chart varies <strong>one parameter at a time</strong> while keeping all others fixed at their optimal values. The 🟢 green shaded zone marks the optimal range. <strong>Steep slopes</strong> = this parameter is critical and sensitive to change. <strong>Flat lines</strong> = less critical, small changes won\'t matter much.</div>', unsafe_allow_html=True)

        sens_params = {
            "Feed_Amount":   (np.linspace(10,20,30),  "kg"),
            "Temperature":   (np.linspace(4, 15,30),  "°C"),
            "pH":            (np.linspace(4, 12,30),  ""),
            "Water_Added":   (np.linspace(2, 20,30),  "L"),
        }

        fig_s = make_subplots(rows=2,cols=2,
            subplot_titles=[f"{FEATURE_LABELS.get(p,'').split(' (')[0]} vs Predicted Yield"
                            for p in sens_params],
            vertical_spacing=0.18)
        pos  = [(1,1),(1,2),(2,1),(2,2)]
        clrs = ["#2D6A4F","#E63946","#457B9D","#F9A825"]

        for idx,(param,(values,unit)) in enumerate(sens_params.items()):
            preds = []
            for v in values:
                cond = dict(best); cond[param] = float(v)
                preds.append(predict_single(bundle, cond))
            r,c = pos[idx]
            lo = GUIDANCE_THRESHOLDS.get(param,{}).get("optimal_min")
            hi = GUIDANCE_THRESHOLDS.get(param,{}).get("optimal_max")
            if lo is not None and hi is not None:
                fig_s.add_vrect(x0=lo,x1=hi,
                    fillcolor="rgba(82,183,136,.2)",line_width=0,row=r,col=c)
            fig_s.add_trace(go.Scatter(x=values,y=preds,mode="lines+markers",
                line=dict(color=clrs[idx],width=2.5),
                marker=dict(size=5,color=clrs[idx]),
                showlegend=False),row=r,col=c)

        fig_s.update_layout(height=500,plot_bgcolor="#ffffff",paper_bgcolor="#f0f7f2",
            font=dict(color="#1B4332",family="Inter"),margin=dict(t=55,b=10))
        fig_s.update_yaxes(title_text="Predicted yield (kg)",gridcolor="#e0ede6",
                           tickfont=dict(color="#1B4332"))
        fig_s.update_xaxes(gridcolor="#e0ede6",tickfont=dict(color="#1B4332"))
        for ann in fig_s.layout.annotations: ann.font.color = "#1B4332"
        st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("""
        <div class="box-info">
        <strong>How to read these charts:</strong><br>
        • 🟢 <strong>Green zone</strong> = the optimal range recommended above<br>
        • A <strong>peak inside or near the green zone</strong> confirms the recommendation is consistent with the model<br>
        • <strong>Steep drop-off outside the zone</strong> = this parameter is very sensitive — stay within range<br>
        • <strong>Mostly flat line</strong> = the model found less variation due to this parameter — other factors matter more
        </div>""", unsafe_allow_html=True)
