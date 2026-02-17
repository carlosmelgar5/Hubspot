import pandas as pd 
import numpy as np

df = pd.read_csv("/Users/Apple/Desktop/Hubspot/data/processed/nps.csv") # open file

print(df.head()) #top of data
print(df.tail()) #bottom of data

print(df.info()) # column names and data type


#Numerical: score, Numerical (but identifier — treat as categorical in analysis):response_id

#Categorical: taxonomy_type, page_url, Text (qualitative / NLP type):response_text, translated_text, Date (currently categorical but should be converted): date

import pandas as pd
import numpy as np

# ── Load Data ──
df = pd.read_csv("/Users/Apple/Desktop/Hubspot/data/processed/nps.csv")

# ── Data Overview ──
print("=" * 40)
print("DATA OVERVIEW")
print("=" * 40)
print(f"Total Records:      {len(df)}")
print(f"Number of Features: {df.shape[1]}")
print(f"Columns:            {list(df.columns)}")
print(f"Date Range:         {df[df.columns[df.columns.str.contains('date', case=False)]].min().values[0]} – {df[df.columns[df.columns.str.contains('date', case=False)]].max().values[0]}")
print()

# ── Find the NPS score column (adjust if needed) ──
# Try to auto-detect the score column
score_col = [c for c in df.columns if 'score' in c.lower() or 'nps' in c.lower() or 'rating' in c.lower()]
print(f"Detected score column(s): {score_col}")
print(">>> If wrong, change 'score_col' below manually <<<")
print()

# Set this to your actual NPS score column name
score_col = score_col[0] if score_col else df.columns[0]  # fallback to first column

# ── Score Distribution ──
promoters = df[df[score_col] >= 9]
passives = df[(df[score_col] >= 7) & (df[score_col] <= 8)]
detractors = df[df[score_col] <= 6]

total = len(df)
prom_pct = len(promoters) / total * 100
pass_pct = len(passives) / total * 100
det_pct = len(detractors) / total * 100
nps = prom_pct - det_pct

print("=" * 40)
print("SCORE DISTRIBUTION")
print("=" * 40)
print(f"Promoters (9-10):   {len(promoters):>6}  ({prom_pct:.1f}%)")
print(f"Passives (7-8):     {len(passives):>6}  ({pass_pct:.1f}%)")
print(f"Detractors (0-6):   {len(detractors):>6}  ({det_pct:.1f}%)")
print(f"Overall NPS:        {nps:.1f}")
print()

# ── Descriptive Statistics ──
print("=" * 40)
print("DESCRIPTIVE STATISTICS")
print("=" * 40)
print(f"Mean:               {df[score_col].mean():.2f}")
print(f"Median:             {df[score_col].median():.2f}")
print(f"Std. Deviation:     {df[score_col].std():.2f}")
print(f"Min / Max:          {df[score_col].min()} / {df[score_col].max()}")