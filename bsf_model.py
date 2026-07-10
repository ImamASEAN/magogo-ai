"""
MAGOGO — BSF Farming Optimisation | Core ML Pipeline
"""

import warnings, itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

FEATURES = [
    "Feed_Amount","Cycle_Days","Bulking_Agent","Water_Added",
    "pH","Temperature","Reactor_Temperature","Frass_Mass",
]
FEATURE_LABELS = {
    "Feed_Amount":"Feed Amount (kg)","Cycle_Days":"Cycle Duration (days)",
    "Bulking_Agent":"Bulking Agent (kg)","Water_Added":"Water Added (L)",
    "pH":"pH Level","Temperature":"Ambient Temp (°C)",
    "Reactor_Temperature":"Reactor Temp (°C)","Frass_Mass":"Frass Mass (kg)",
}

# Grid: 6×6×5×5×4 = 3600 combos, ~0.03s with batch prediction
PARAM_RANGES = {
    "Feed_Amount":   np.array([10.0,11.0,12.0,13.0,15.0,18.0,20.0]),
    "Temperature":   np.array([5.0,6.0,7.0,8.0,9.0,10.0]),
    "pH":            np.array([6.0,7.0,8.0,9.0,10.0]),
    "Water_Added":   np.array([5.0,7.5,10.0,12.5,15.0]),
    "Bulking_Agent": np.array([0.5,1.0,1.5,2.0]),
}

GUIDANCE_THRESHOLDS = {
    "Feed_Amount":   {"optimal_min":12.0,"optimal_max":13.0,"unit":"kg",
        "low_tip":"Increase feed to 12–13 kg. Low feed limits how much biomass larvae can produce.",
        "high_tip":"Reduce feed below 13.5 kg. Excess feed causes anaerobic fermentation.",
    },
    "Temperature":   {"optimal_min":6.0,"optimal_max":8.5,"unit":"°C",
        "low_tip":"Ambient temp is low. Insulate your reactor or use a heating mat.",
        "high_tip":"Temperature is high. Add shade or ventilation to avoid heat stress on larvae.",
    },
    "pH":            {"optimal_min":8.0,"optimal_max":10.0,"unit":"",
        "low_tip":"pH too acidic. Add lime or wood ash to raise pH toward 8–10.",
        "high_tip":"pH too alkaline. Reduce lime and add more carbon-rich bulking agent.",
    },
    "Water_Added":   {"optimal_min":5.0,"optimal_max":10.0,"unit":"L",
        "low_tip":"Substrate is too dry. Add 5–10 L of water to help larvae feed.",
        "high_tip":"Too much water. Reduce to 5–10 L and add bulking agent to improve drainage.",
    },
    "Bulking_Agent": {"optimal_min":0.5,"optimal_max":1.0,"unit":"kg",
        "low_tip":"Add more bulking agent (wood chips or cardboard) to improve aeration.",
        "high_tip":"Too much bulking agent dilutes nutrients. Keep to 0.5–1.0 kg.",
    },
    "Cycle_Days":    {"optimal_min":180,"optimal_max":300,"unit":"days",
        "low_tip":"Cycle too short — larvae may not have reached prepupal stage yet. Target 180–300 days.",
        "high_tip":"Very long cycle. Harvest at prepupal stage (days 180–300) before larvae lose mass.",
    },
}

WHY_OPTIMAL = {
    "Feed_Amount":   "Feed is the #1 yield driver in your data. The 12–13 kg range gives larvae enough organic matter without causing oxygen depletion in the substrate.",
    "Temperature":   "Your highest-yield cycles occurred at ambient temps of 6–8.5°C. The reactor naturally warms the substrate above ambient, so this range is sufficient.",
    "pH":            "A slightly alkaline substrate (pH 8–10) inhibits competing microbes and supports efficient larval digestion of organic waste.",
    "Water_Added":   "Adequate moisture (5–10 L) keeps the substrate soft enough for larvae to feed while maintaining airflow through the bulking agent.",
    "Bulking_Agent": "Wood chips or cardboard (0.5–1 kg) create air pockets that prevent anaerobic dead zones — critical for healthy, fast-growing larvae.",
    "Cycle_Days":    "Harvest at the prepupal stage (days 180–300) when larvae reach maximum biomass. Harvesting too early or too late reduces yield.",
    "Reactor_Temperature":"Fixed at training median. Insulate the reactor to maintain this internal temperature regardless of ambient conditions.",
    "Frass_Mass":    "Frass output is a result of the process, not an input you control directly. Used here as a reference value only.",
}

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, sep=";", decimal=",", na_values=["", " "])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.replace(["N/A","NA","n/a","na","-"], np.nan, inplace=True)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)

def _pipe(est):
    return Pipeline([("imp",SimpleImputer(strategy="median")),
                     ("scl",StandardScaler()),("mdl",est)])

def _cv(pipe, X, y, cv):
    r = cross_validate(pipe, X, y, cv=cv,
                       scoring=["neg_mean_absolute_error","r2"])
    return {"mae_mean":-r["test_neg_mean_absolute_error"].mean(),
            "mae_std": r["test_neg_mean_absolute_error"].std(),
            "r2_mean": r["test_r2"].mean(),
            "r2_std":  r["test_r2"].std()}

def train_models(df):
    hdf   = df.dropna(subset=["Harvest_Mass"]).copy()
    feats = [f for f in FEATURES if f in hdf.columns]
    imp   = SimpleImputer(strategy="median")
    X     = pd.DataFrame(imp.fit_transform(hdf[feats]), columns=feats)
    y     = hdf["Harvest_Mass"].values
    
    # Feature Engineering: Tambahkan polynomial features dan interaksi
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.compose import ColumnTransformer
    
    # Buat fitur tambahan: rasio dan interaksi
    X_enhanced = X.copy()
    
    # Tambahkan rasio Feed/Temperature sebagai fitur baru
    if "Feed_Amount" in X.columns and "Temperature" in X.columns:
        X_enhanced["Feed_Temp_Ratio"] = X_enhanced["Feed_Amount"] / (X_enhanced["Temperature"] + 1)
    
    # Tambahkan interaksi Feed * Water
    if "Feed_Amount" in X.columns and "Water_Added" in X.columns:
        X_enhanced["Feed_Water_Interact"] = X_enhanced["Feed_Amount"] * X_enhanced["Water_Added"]
    
    # Tambahkan kuadrat dari fitur penting
    if "Feed_Amount" in X.columns:
        X_enhanced["Feed_Squared"] = X_enhanced["Feed_Amount"] ** 2
    if "Temperature" in X.columns:
        X_enhanced["Temp_Squared"] = X_enhanced["Temperature"] ** 2
    
    # Split data untuk evaluasi (80% train, 20% test)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)
    
    cv    = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Model candidates dengan hyperparameter yang dioptimalkan
    cands = {
        "Gradient Boosting": _pipe(GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )),
        "Random Forest": _pipe(RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )),
        "Ridge Regression": _pipe(Ridge(alpha=10.0)),
    }
    
    rows = [{"Model":n,**_cv(p,X_enhanced,y,cv)} for n,p in cands.items()]
    comp = pd.DataFrame(rows)
    best_name = comp.loc[comp["mae_mean"].idxmin(),"Model"]
    best_pipe = cands[best_name]
    best_pipe.fit(X_enhanced,y)
    
    # Fit model terbaik pada training data dan evaluate pada test data
    best_pipe.fit(X_train, y_train)
    y_pred_test = best_pipe.predict(X_test)
    
    rf = cands["Random Forest"]; rf.fit(X_enhanced,y)
    fi = pd.DataFrame({"Feature":X_enhanced.columns,
                       "Label":[FEATURE_LABELS.get(f,f) for f in X_enhanced.columns],
                       "Importance":rf.named_steps["mdl"].feature_importances_
                       }).sort_values("Importance",ascending=False).reset_index(drop=True)
    return {"best_model":best_pipe,"best_name":best_name,"comparison":comp,
            "feature_imp":fi,"X":X_enhanced,"y":y,"imputer":imp,"features_used":list(X_enhanced.columns),
            "harvest_df":hdf,
            "high_yield_threshold":float(np.percentile(y,75)),
            "low_yield_threshold": float(np.percentile(y,25)),
            "median_yield":        float(np.median(y)),
            "y_test": y_test,
            "y_pred_test": y_pred_test}

def predict_single(bundle, user_inputs):
    feats = bundle["features_used"]
    row   = pd.DataFrame([{f:user_inputs.get(f,np.nan) for f in feats}])
    row_i = pd.DataFrame(bundle["imputer"].transform(row),columns=feats)
    return float(bundle["best_model"].predict(row_i)[0])

def optimise_conditions(bundle):
    """Batch-predict all ~3600 grid combinations — runs in <1 second."""
    feats   = bundle["features_used"]
    medians = bundle["X"].median().to_dict()
    keys    = [k for k in PARAM_RANGES if k in feats]
    combos  = list(itertools.product(*[PARAM_RANGES[k] for k in keys]))

    rows = []
    for combo in combos:
        cond = {k:float(v) for k,v in zip(keys,combo)}
        for f in feats:
            if f not in cond:
                cond[f] = float(medians.get(f,np.nan))
        rows.append([cond[f] for f in feats])

    X_all  = pd.DataFrame(rows, columns=feats)
    X_imp  = pd.DataFrame(bundle["imputer"].transform(X_all), columns=feats)
    preds  = bundle["best_model"].predict(X_imp)
    best_i = int(np.argmax(preds))

    best_cond = {k:float(v) for k,v in zip(keys,combos[best_i])}
    for f in feats:
        if f not in best_cond:
            best_cond[f] = float(medians.get(f,np.nan))
    best_cond["Predicted_Harvest_Mass"] = float(preds[best_i])
    return best_cond

def generate_guidance(user_inputs, bundle):
    tips = []
    for param, rules in GUIDANCE_THRESHOLDS.items():
        if param not in user_inputs:
            continue
        val = user_inputs[param]
        lo, hi = rules["optimal_min"], rules["optimal_max"]
        if val < lo:
            status, tip_text = "below", rules["low_tip"]
            priority = "high" if (lo-val)/(hi-lo+1e-6) > 0.5 else "medium"
        elif val > hi:
            status, tip_text = "above", rules["high_tip"]
            priority = "high" if (val-hi)/(hi-lo+1e-6) > 0.5 else "medium"
        else:
            status   = "optimal"
            tip_text = f"Within the ideal range ({lo}–{hi}{rules['unit']}) — keep it here."
            priority = "ok"
        tips.append({"parameter":param,"label":FEATURE_LABELS.get(param,param),
                     "current_value":val,"unit":rules["unit"],
                     "optimal_min":lo,"optimal_max":hi,
                     "status":status,"tip":tip_text,"priority":priority})
    tips.sort(key=lambda x:{"high":0,"medium":1,"ok":2}[x["priority"]])
    return tips

def yield_interpretation(predicted, bundle):
    median = bundle["median_yield"]
    high_t = bundle["high_yield_threshold"]
    low_t  = bundle["low_yield_threshold"]
    delta  = predicted - median
    pct    = delta / median * 100
    if predicted >= high_t:
        level,emoji,msg = "Excellent","🟢","High-yield scenario — conditions are well-suited for BSF production."
    elif predicted >= median:
        level,emoji,msg = "Good","🟡","Above average. Small adjustments could push this into the high range."
    elif predicted >= low_t:
        level,emoji,msg = "Below Average","🟠","Below historical average. Follow the guidance tips below to improve."
    else:
        level,emoji,msg = "Poor","🔴","Low predicted yield. Apply high-priority tips for the biggest gains."
    return {"level":level,"emoji":emoji,"message":msg,
            "delta_kg":delta,"delta_pct":pct,
            "median":median,"high_threshold":high_t,"low_threshold":low_t}
