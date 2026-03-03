import duckdb
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# ── Paths
BASE_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = BASE_DIR / "notebooks" / "Database" / "hubspot.db"
MODEL_PATH  = BASE_DIR / "models" / "frustration_model.pkl"
REPORT_DIR  = BASE_DIR / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully")

# ── Connect
conn = duckdb.connect(str(DB_PATH))
print("Connected to:", DB_PATH)

from sklearn.preprocessing import LabelEncoder

# ── Load frustration signals
frustration = conn.execute("SELECT * FROM frustration_signals").df()
tickets     = conn.execute("SELECT * FROM support_tickets").df()
csat        = conn.execute("SELECT * FROM csat").df()
nps         = conn.execute("SELECT * FROM nps").df()

# ── Label
frustration["label"] = 1
tickets["label"]     = 1
csat["label"] = csat["Score"].apply(lambda x: 1 if x <= 3 else 0)
nps["label"]  = nps["Score"].apply(lambda x: 1 if x <= 6 else 0)

fs  = frustration[["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME", "COUNTRY", "label"]]
tk  = tickets[["Product Area", "Roadblock", "label", "Portal ID"]]
cs  = csat[["Taxonomy Type", "label", "Portal ID"]]
np_ = nps[["Taxonomy Type", "label"]]

# ── Standardise portal ID column name
tk = tk.rename(columns={"Portal ID": "PORTAL_ID"})
cs = cs.rename(columns={"Portal ID": "PORTAL_ID"})

# ── Add empty PORTAL_ID to tables that don't have it
fs["PORTAL_ID"]  = "unknown"
np_["PORTAL_ID"] = "unknown"

# ── Combine
df = pd.concat([fs, tk, cs, np_], ignore_index=True)
df = df.fillna("unknown")

# ── Encode
le = LabelEncoder()
cat_cols = ["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME",
            "COUNTRY", "Product Area", "Roadblock", "Taxonomy Type"]
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

for col in df.columns:
    if col not in ["label", "PORTAL_ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

print("Data prepared:", df.shape)
print("Accounts found:", df["PORTAL_ID"].nunique())

# ── Score every row
X_score = df.drop(columns=["label", "PORTAL_ID"])
df["frustration_score"] = model.predict_proba(X_score)[:, 1]

# ── Group by account and calculate health score
health = (
    df[df["PORTAL_ID"] != "unknown"]
    .groupby("PORTAL_ID")
    .agg(
        total_signals    = ("frustration_score", "count"),
        avg_frustration  = ("frustration_score", "mean"),
        max_frustration  = ("frustration_score", "max"),
    )
    .reset_index()
)

# ── Rank accounts — highest avg frustration at top
health = health.sort_values("avg_frustration", ascending=False)
health["rank"] = range(1, len(health) + 1)

# ── Flag high risk accounts
health["risk_level"] = health["avg_frustration"].apply(
    lambda x: "HIGH" if x >= 0.75 else ("MEDIUM" if x >= 0.50 else "LOW")
)

print("── Weekly Health Score Preview ──")
print(health.head(10))
print(f"\nHigh risk accounts:   {len(health[health['risk_level'] == 'HIGH'])}")
print(f"Medium risk accounts: {len(health[health['risk_level'] == 'MEDIUM'])}")
print(f"Low risk accounts:    {len(health[health['risk_level'] == 'LOW'])}")

# ── Save report
today = datetime.today().strftime("%Y-%m-%d")
report_path = REPORT_DIR / f"health_scores_{today}.csv"
health.to_csv(report_path, index=False)
print(f"\nReport saved to: {report_path}")
print(f"Total accounts scored: {len(health)}")
print(f"\nDone. Run this every Monday morning.")