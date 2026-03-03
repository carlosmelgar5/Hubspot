import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "notebooks" / "Database" / "hubspot.db"
OUT_DIR  = BASE_DIR / "outputs" / "reports"

# ── Connect
conn = duckdb.connect(str(DB_PATH))
print("Connected to:", DB_PATH)

# ── Load each table
frustration = conn.execute("SELECT * FROM frustration_signals").df()
tickets     = conn.execute("SELECT * FROM support_tickets").df()
csat        = conn.execute("SELECT * FROM csat").df()
nps         = conn.execute("SELECT * FROM nps").df()

# ── Quick sanity check
print("frustration_signals:", frustration.shape)
print("support_tickets:    ", tickets.shape)
print("csat:               ", csat.shape)
print("nps:                ", nps.shape)


# ── Label each source
frustration["label"] = 1
tickets["label"]     = 1

csat["label"] = csat["Score"].apply(lambda x: 1 if x <= 3 else 0)
nps["label"]  = nps["Score"].apply(lambda x: 1 if x <= 6 else 0)

# ── Keep only what we need
fs = frustration[["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME", "COUNTRY", "label"]]
tk = tickets[["Product Area", "Roadblock", "label"]]
cs  = csat[["Taxonomy Type", "label"]]
np_ = nps[["Taxonomy Type", "label"]]

# ── Combine
from sklearn.preprocessing import LabelEncoder

df = pd.concat([fs, tk, cs, np_], ignore_index=True)

print("Training dataset shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts())


# ── Fill nulls
df = df.fillna("unknown")

# ── Encode categorical columns into numbers
cat_cols = ["SIGNAL_NAME", "PAGE_CATEGORY", "DEPLOYABLE_NAME",
            "COUNTRY", "Product Area", "Roadblock",
            "Survey Name", "Taxonomy Type", "Page URL"]

le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# ── Force ALL remaining columns to numeric
for col in df.columns:
    if col != "label":
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── Confirm no object columns remain
obj_cols = df.select_dtypes(include="object").columns.tolist()
print("Object columns remaining:", obj_cols)

from sklearn.utils import resample

frustrated     = df[df["label"] == 1].sample(n=500000, random_state=42)
not_frustrated = df[df["label"] == 0]

df_balanced = pd.concat([frustrated, not_frustrated], ignore_index=True)

print("Balanced distribution:")
print(df_balanced["label"].value_counts())

X = df_balanced.drop(columns=["label"])
y = df_balanced["label"]

# ── Calculate scale weight
scale = round(y.value_counts()[1] / y.value_counts()[0])
print("scale_pos_weight:", scale)


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle

# ── Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Test rows:    ", X_test.shape[0])

# ── Define model
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

# ── Train
print("Training model...")
model.fit(X_train, y_train)
print("Done.")

# ── Save model
model_path = BASE_DIR / "models" / "frustration_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print("Model saved to:", model_path)