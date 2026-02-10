import pandas as pd
from pathlib import Path

# Load cleaned data
processed_dir = Path("data/processed")

nps = pd.read_excel(processed_dir / "nps_clean.xlsx")
csat = pd.read_excel(processed_dir / "csat_clean.xlsx")
tickets = pd.read_excel(processed_dir / "support_tickets_clean.xlsx")

# Data Profile Function
def data_profile(df, name="Dataset"):
    print(f"\n{'='*60}")
    print(f"  DATA PROFILE: {name}")
    print(f"{'='*60}")

    # Dataset description
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Time period covered
    date_cols = df.select_dtypes(include=["datetime64"]).columns
    for col in date_cols:
        min_date = df[col].min()
        max_date = df[col].max()
        print(f"Time range ({col}): {min_date} → {max_date}")

    # Per-column breakdown
    print(f"\n{'─'*60}")
    print(f"{'Column':<40} {'Dtype':<15} {'Nulls':<12} {'Distinct'}")
    print(f"{'─'*60}")

    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        null_pct = f"{nulls} ({nulls/len(df)*100:.1f}%)"
        distinct = df[col].nunique()
        print(f"{col:<40} {dtype:<15} {null_pct:<12} {distinct}")

    # Obvious issues
    print(f"\n{'─'*60}")
    print("ISSUES:")

    issues_found = False

    # High null columns (>50%)
    for col in df.columns:
        null_pct = df[col].isnull().sum() / len(df) * 100
        if null_pct > 50:
            print(f"  ⚠ {col}: {null_pct:.1f}% null")
            issues_found = True

    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"  ⚠ {dupes:,} duplicate rows")
        issues_found = True

    # All-null columns
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        print(f"  ⚠ Entirely empty columns: {empty_cols}")
        issues_found = True

    if not issues_found:
        print("  ✓ No major issues found")

    print(f"\n{'='*60}\n")

# Run profiles

if __name__ == "__main__":
    data_profile(nps, "NPS Data")
    data_profile(csat, "CSAT Data")
    data_profile(tickets, "Support Tickets")
    