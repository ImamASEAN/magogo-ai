"""
================================================================
  MAGOGO — BSF FARMING OPTIMISATION  |  Core ML Pipeline
================================================================
  This module handles all machine-learning work:
    • Data loading & cleaning
    • Model training (Random Forest, Gradient Boosting, Linear)
    • Cross-validation & model comparison
    • Feature importance
    • Yield prediction
    • Condition optimisation (grid search over farming parameters)

  Import this in app.py — it has no UI code of its own.
================================================================
"""

import os
import warnings
import itertools

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model  import Ridge
from sklearn.model_selection import cross_validate, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─── Feature & label maps ────────────────────────────────────────────────────
FEATURES = [
    "Feed_Amount",
    "Cycle_Days",
    "Bulking_Agent",
    "Water_Added",
    "pH",
    "Temperature",
    "Reactor_Temperature",
    "Frass_Mass",
]

FEATURE_LABELS = {
    "Feed_Amount":         "Feed Amount (kg)",
    "Cycle_Days":          "Cycle Duration (days)",
    "Bulking_Agent":       "Bulking Agent (kg)",
    "Water_Added":         "Water Added (L)",
    "pH":                  "pH Level",
    "Temperature":         "Ambient Temp (°C)",
    "Reactor_Temperature": "Reactor Temp (°C)",
    "Frass_Mass":          "Frass Mass (kg)",
}

# ─── Data loading ────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load main_dataset.csv, handle European decimal commas,
    semicolon separators, and missing values.
    Returns a clean DataFrame with numeric columns.
    """
    df = pd.read_csv(
        csv_path,
        sep=";",
        decimal=",",
        na_values=["", " "],
    )

    # Drop stray unnamed columns added by Excel exports
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # Replace common missing-value tokens
    df.replace(["N/A", "NA", "n/a", "na", "-"], np.nan, inplace=True)

    # Cast all columns except Date to numeric
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse dates so we can sort / plot time-series
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ─── Model building helpers ──────────────────────────────────────────────────

def _make_pipeline(estimator):
    """Wrap an estimator in impute → scale → model pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])


def _cv_scores(pipeline, X, y, cv):
    """Return dict with mean/std for MAE and R²."""
    res = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["neg_mean_absolute_error", "r2"],
        return_train_score=False,
    )
    return {
        "mae_mean": -res["test_neg_mean_absolute_error"].mean(),
        "mae_std":   res["test_neg_mean_absolute_error"].std(),
        "r2_mean":   res["test_r2"].mean(),
        "r2_std":    res["test_r2"].std(),
    }


# ─── Main training function ──────────────────────────────────────────────────

def train_models(df: pd.DataFrame):
    """
    Train three models with cross-validation, select the best,
    and return a results bundle used by app.py.

    Returns
    -------
    dict with keys:
        best_model      – fitted sklearn pipeline (best performer)
        best_name       – str label of best model
        comparison      – pd.DataFrame of CV scores per model
        feature_imp     – pd.DataFrame of feature importances
        X, y            – full feature matrix / target (post-imputation)
        imputer         – fitted SimpleImputer (for prediction demo)
        features_used   – list of feature names actually present
    """
    # ── Prepare data ──────────────────────────────────────────
    harvest_df = df.dropna(subset=["Harvest_Mass"]).copy()
    features_used = [f for f in FEATURES if f in harvest_df.columns]

    X_raw = harvest_df[features_used]
    y     = harvest_df["Harvest_Mass"].values

    # Impute once for feature-importance display (RF needs array)
    global_imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        global_imputer.fit_transform(X_raw),
        columns=features_used,
    )

    # ── Cross-validation ──────────────────────────────────────
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "Random Forest": _make_pipeline(
            RandomForestRegressor(
                n_estimators=400, max_features="sqrt",
                min_samples_leaf=1, random_state=42, n_jobs=-1,
            )
        ),
        "Gradient Boosting": _make_pipeline(
            GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=3, random_state=42,
            )
        ),
        "Ridge Regression": _make_pipeline(
            Ridge(alpha=1.0)
        ),
    }

    rows = []
    for name, pipe in candidates.items():
        scores = _cv_scores(pipe, X_imputed, y, cv)
        rows.append({"Model": name, **scores})

    comparison = pd.DataFrame(rows)

    # ── Select best model by lowest MAE ───────────────────────
    best_idx  = comparison["mae_mean"].idxmin()
    best_name = comparison.loc[best_idx, "Model"]
    best_pipe = candidates[best_name]
    best_pipe.fit(X_imputed, y)   # refit on full data

    # ── Feature importance (RF only; Gradient Boosting fallback) ──
    rf_pipe = candidates["Random Forest"]
    rf_pipe.fit(X_imputed, y)
    importances = rf_pipe.named_steps["model"].feature_importances_

    feature_imp = (
        pd.DataFrame({
            "Feature":    features_used,
            "Label":      [FEATURE_LABELS.get(f, f) for f in features_used],
            "Importance": importances,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "best_model":    best_pipe,
        "best_name":     best_name,
        "comparison":    comparison,
        "feature_imp":   feature_imp,
        "X":             X_imputed,
        "y":             y,
        "imputer":       global_imputer,
        "features_used": features_used,
        "harvest_df":    harvest_df,
    }


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_single(model_bundle: dict, user_inputs: dict) -> float:
    """
    Predict Harvest_Mass for a single set of user-supplied conditions.

    Parameters
    ----------
    model_bundle : output of train_models()
    user_inputs  : dict  {feature_name: value}  – missing keys → NaN → imputed

    Returns
    -------
    float : predicted harvest mass in kg
    """
    features_used = model_bundle["features_used"]
    row = pd.DataFrame([{f: user_inputs.get(f, np.nan) for f in features_used}])

    # Re-impute with the training-time imputer
    row_imputed = pd.DataFrame(
        model_bundle["imputer"].transform(row),
        columns=features_used,
    )
    return float(model_bundle["best_model"].predict(row_imputed)[0])


# ─── Optimisation ────────────────────────────────────────────────────────────

def optimise_conditions(model_bundle: dict) -> dict:
    """
    Grid-search over realistic farming parameter ranges.
    Returns the combination that maximises predicted Harvest_Mass.

    Search space (coarse grid — fast enough for a demo):
      Feed       : 10–20 kg   step 2.5
      Temperature: 25–35 °C   step 2.5
      pH         : 6–9        step 1
      Water      : 5–15 L     step 2.5
      Bulking    : 0.2–2 kg   step 0.6
    Fixed values (median from training data):
      Cycle_Days, Reactor_Temperature, Frass_Mass
    """
    features_used = model_bundle["features_used"]
    X_train       = model_bundle["X"]

    # Medians for fixed features
    medians = X_train.median().to_dict()

    grid = {
        "Feed_Amount":   np.arange(10,   20.1,  2.5),
        "Temperature":   np.arange(25,   35.1,  2.5),
        "pH":            np.arange(6,     9.1,  1.0),
        "Water_Added":   np.arange(5,    15.1,  2.5),
        "Bulking_Agent": np.arange(0.2,   2.01, 0.6),
    }

    # Build all combinations
    keys   = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    best_pred = -np.inf
    best_cond = {}

    for combo in combos:
        cond = {k: v for k, v in zip(keys, combo)}
        # Fill remaining features with training medians
        for f in features_used:
            if f not in cond:
                cond[f] = medians.get(f, np.nan)

        pred = predict_single(model_bundle, cond)
        if pred > best_pred:
            best_pred = pred
            best_cond = cond

    best_cond["Predicted_Harvest_Mass"] = best_pred
    return best_cond
