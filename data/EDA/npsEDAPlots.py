import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load Data ──
df = pd.read_csv("/Users/Apple/Desktop/Hubspot/data/processed/nps.csv")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')

# ── Helper: Calculate NPS ──
def calc_nps(scores):
    total = len(scores)
    if total == 0:
        return np.nan
    promoters = (scores >= 9).sum() / total * 100
    detractors = (scores <= 6).sum() / total * 100
    return promoters - detractors

# ── Color Palette ──
NAVY = "#1B3A5C"
BLUE = "#2E75B6"
GREEN = "#27AE60"
RED = "#E74C3C"
ORANGE = "#F39C12"
GRAY = "#95A5A6"
LIGHT_BG = "#F8F9FA"

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.facecolor': LIGHT_BG,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#CCCCCC',
})


# ======================================================================
# CHART 1: NPS OVER TIME (MONTHLY)
# ======================================================================
# WHAT IT SHOWS:
# The overall NPS score calculated per month as a line chart.
# This reveals whether customer sentiment is improving, declining,
# or staying flat over time. Look for:
#   - Upward/downward trends
#   - Seasonal patterns (e.g. dips in certain months)
#   - Sudden drops that may correlate with product changes or incidents
# ======================================================================

monthly_nps = df.groupby('month')['score'].apply(calc_nps).reset_index()
monthly_nps.columns = ['month', 'nps']
monthly_nps['month_str'] = monthly_nps['month'].astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly_nps['month_str'], monthly_nps['nps'], color=BLUE, marker='o', linewidth=2.5, markersize=6)
ax.axhline(y=0, color=RED, linestyle='--', alpha=0.5, label='NPS = 0 (Neutral)')
ax.fill_between(monthly_nps['month_str'], monthly_nps['nps'], 0,
                where=monthly_nps['nps'] >= 0, alpha=0.1, color=GREEN)
ax.fill_between(monthly_nps['month_str'], monthly_nps['nps'], 0,
                where=monthly_nps['nps'] < 0, alpha=0.1, color=RED)
ax.set_title('NPS Over Time (Monthly)', fontsize=14, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel('Month', fontsize=11, color=NAVY)
ax.set_ylabel('NPS Score', fontsize=11, color=NAVY)
ax.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/Users/Apple/Desktop/Hubspot/data/EDA/plots/chart1_nps_over_time.png", dpi=150, bbox_inches='tight')
plt.show()
print("Chart 1 saved.\n")


# ======================================================================
# CHART 2: NPS BY TAXONOMY TYPE
# ======================================================================
# WHAT IT SHOWS:
# Compares NPS across feedback categories: Usability, Functionality,
# Performance, Reliability, and General. This tells you which areas
# of your product customers are happiest/unhappiest with.
# Look for:
#   - Which category has the lowest NPS (priority area for improvement)
#   - Which category has the highest NPS (your strength)
#   - Big gaps between categories signal uneven product experience
# NOTE: NA values are excluded since they don't represent a category.
# ======================================================================

df_tax = df[df['taxonomy_type'].notna() & (df['taxonomy_type'] != 'NA')].copy()
tax_nps = df_tax.groupby('taxonomy_type')['score'].apply(calc_nps).sort_values(ascending=True).reset_index()
tax_nps.columns = ['taxonomy_type', 'nps']
tax_counts = df_tax.groupby('taxonomy_type')['score'].count().reset_index()
tax_counts.columns = ['taxonomy_type', 'count']
tax_nps = tax_nps.merge(tax_counts, on='taxonomy_type')

colors = [GREEN if x >= 0 else RED for x in tax_nps['nps']]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(tax_nps['taxonomy_type'], tax_nps['nps'], color=colors, edgecolor='white', height=0.55)
ax.axvline(x=0, color='black', linewidth=0.8)

# Add count labels with padding to avoid overlap
for i, (nps_val, count) in enumerate(zip(tax_nps['nps'], tax_nps['count'])):
    if nps_val >= 0:
        ax.text(nps_val + 1.5, i, f'{nps_val:.0f}  (n={count})', va='center', ha='left', fontsize=10, color=NAVY, fontweight='bold')
    else:
        ax.text(nps_val - 1.5, i, f'{nps_val:.0f}  (n={count})', va='center', ha='right', fontsize=10, color=NAVY, fontweight='bold')

# Add x-axis padding so labels don't get cut off or overlap y-axis
x_min = tax_nps['nps'].min()
x_max = tax_nps['nps'].max()
ax.set_xlim(x_min - 25, x_max + 25)

ax.set_title('NPS by Taxonomy Type', fontsize=16, fontweight='bold', color=NAVY, pad=20)
ax.set_xlabel('NPS Score', fontsize=12, color=NAVY)
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.savefig("/Users/Apple/Desktop/Hubspot/data/EDA/plots/chart2_nps_by_taxonomy.png", dpi=150, bbox_inches='tight')
plt.show()
print("Chart 2 saved.\n")


# ======================================================================
# CHART 3: RESPONSE VOLUME OVER TIME
# ======================================================================
# WHAT IT SHOWS:
# The number of survey responses received each month. This helps you
# understand data reliability and engagement patterns. Look for:
#   - Low-volume months (NPS in those months may be less reliable)
#   - Spikes that may coincide with survey campaigns or product launches
#   - Declining volume could mean survey fatigue or delivery issues
# ======================================================================

monthly_vol = df.groupby('month')['response_id'].count().reset_index()
monthly_vol.columns = ['month', 'responses']
monthly_vol['month_str'] = monthly_vol['month'].astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(monthly_vol['month_str'], monthly_vol['responses'], color=BLUE, edgecolor='white', width=0.7)

# Add value labels on bars
for i, v in enumerate(monthly_vol['responses']):
    ax.text(i, v + max(monthly_vol['responses']) * 0.02, str(v), ha='center', va='bottom', fontsize=9, color=NAVY)

ax.set_title('Response Volume Over Time', fontsize=14, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel('Month', fontsize=11, color=NAVY)
ax.set_ylabel('Number of Responses', fontsize=11, color=NAVY)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/Users/Apple/Desktop/Hubspot/data/EDA/plots/chart3_response_volume.png", dpi=150, bbox_inches='tight')
plt.show()
print("Chart 3 saved.\n")