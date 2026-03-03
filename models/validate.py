import duckdb
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# ── Paths
BASE_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = BASE_DIR / "notebooks" / "Database" / "hubspot.db"
MODEL_PATH  = BASE_DIR / "models" / "frustration_model.pkl"
OUT_DIR     = BASE_DIR / "outputs" / "reports"

# ── Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")

# ── Connect to DB
conn = duckdb.connect(str(DB_PATH))
print("Connected to:", DB_PATH)

# ── Reload all 4 tables (same as train.py)
frustration = conn.execute("SELECT * FROM frustration_signals").df()
tickets     = conn.execute("SELECT * FROM support_tickets").df()
csat        = conn.execute("SELECT * FROM csat").df()
nps         = conn.execute("SELECT * FROM nps").df()

# ── Label
frustration["label"] = 1
tickets["label"]     = 1
csat["label"] = csat["Score"].apply(lambda x: 1 if x <= 3 else 0)
nps["label"]  = nps["Score"].apply(lambda x: 1 if x <= 6 else 0)

# ── Trim columns
fs  = frustration[["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME", "COUNTRY", "label"]]
tk  = tickets[["Product Area", "Roadblock", "label"]]
cs  = csat[["Taxonomy Type", "label"]]
np_ = nps[["Taxonomy Type", "label"]]

# ── Combine and clean
df = pd.concat([fs, tk, cs, np_], ignore_index=True)
df = df.fillna("unknown")

le = LabelEncoder()
cat_cols = ["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME",
            "COUNTRY", "Product Area", "Roadblock",
            "Survey Name", "Taxonomy Type", "Page URL"]
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── Split same way as train.py
from sklearn.model_selection import train_test_split
X = df.drop(columns=["label"])
y = df["label"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Predict
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Predictions done")
print("Test set size:", len(y_test))


# ── Classification report
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Not Frustrated", "Frustrated"]))

# ── Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("── Confusion Matrix ──")
print(f"True Negatives  (correctly said not frustrated): {cm[0][0]}")
print(f"False Positives (wrongly flagged as frustrated): {cm[0][1]}")
print(f"False Negatives (missed frustrated users):       {cm[1][0]}")
print(f"True Positives  (correctly caught frustrated):   {cm[1][1]}")

# ── ROC AUC
auc = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC Score: {auc:.4f}")
print("(1.0 = perfect, 0.5 = random guessing)")

# ── Feature importance
print("\n── Top Features by Importance ──")
feat_imp = pd.Series(model.feature_importances_, index=X_test.columns)
print(feat_imp.sort_values(ascending=False))

# ══════════════════════════════════════════
# HOLDOUT TEST — Last 2 weeks of fresh data
# ══════════════════════════════════════════

# ── Pull only recent frustration signals
holdout = conn.execute("""
    SELECT * FROM frustration_signals
    WHERE EVENT_TIME >= '2026-01-22'
""").df()

print(f"Holdout rows: {holdout.shape[0]}")

# ── Label and prepare
holdout["label"] = 1

# ── We need some not-frustrated rows too
csat_recent = conn.execute("""
    SELECT * FROM csat
    WHERE Date >= '2026-01-22'
""").df()

csat_recent["label"] = csat_recent["Score"].apply(lambda x: 1 if x <= 3 else 0)

# ── Trim columns
h_fs = holdout[["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME", "COUNTRY", "label"]]
h_cs = csat_recent[["Taxonomy Type", "label"]]

df_holdout = pd.concat([h_fs, h_cs], ignore_index=True)
df_holdout  = df_holdout.fillna("unknown")

# ── Encode
le2 = LabelEncoder()
for col in cat_cols:
    if col in df_holdout.columns:
        df_holdout[col] = le2.fit_transform(df_holdout[col].astype(str))

for col in df_holdout.columns:
    if col != "label":
        df_holdout[col] = pd.to_numeric(df_holdout[col], errors="coerce").fillna(0)

# ── Predict
X_holdout = df_holdout.drop(columns=["label"])

# ── Add missing columns the model expects
for col in ["Product Area", "Roadblock"]:
    if col not in X_holdout.columns:
        X_holdout[col] = 0

# ── Reorder columns to match training
expected_cols = ['SIGNAL_NAME', 'PAGE_CATEGORY', 'DEPLOYABLE_NAME', 
                 'COUNTRY', 'Product Area', 'Roadblock', 'Taxonomy Type']
X_holdout = X_holdout[expected_cols]

y_holdout  = df_holdout["label"]



y_holdout_pred = model.predict(X_holdout)

print("\n── Holdout Classification Report ──")
print(classification_report(y_holdout, y_holdout_pred,
      target_names=["Not Frustrated", "Frustrated"]))