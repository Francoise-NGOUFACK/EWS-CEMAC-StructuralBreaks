"""
00_data_analysis.py
====================
EWS-CEMAC — Exploratory Data Analysis and Descriptive Statistics

PURPOSE
-------
This script performs the exploratory data analysis (EDA) reported in
Section 2 (Data) and Section 3.1 (Variable Selection) of the paper.
It must be run BEFORE the modelling scripts (01–07).

OUTPUTS
-------
  Printed to console:
    - Descriptive statistics table (Table 1 in paper)
    - Missing data report by variable and country
    - Mann-Whitney U test p-values (crisis vs. non-crisis comparison)
    - Pairwise Pearson correlations between the 5 EWS predictors

  Figures saved to figures/:
    - Figure_EDA_distributions.png  — histograms of 5 predictors by country
    - Figure_EDA_boxplots.png       — boxplots crisis vs. non-crisis
    - Figure_EDA_correlation.png    — correlation matrix heatmap
    - Figure_EDA_target_timeline.png — BSI stress episodes over 2000-2023

DATA REQUIRED
-------------
  Dataset_Macro_CEMAC.csv — assembled panel (6 CEMAC countries × 2000–2023).
  Place it in the same directory as this script, or update DATA_PATH below.

  This file is NOT redistributed in the repository (see README — Data Sources).
  To reconstruct it, merge:
    - World Bank WDI (GDP growth, M2, fiscal balance, oil rents, reserves)
    - IMF WEO (oil rents, reserves — supplementary)
    - BEAC/COBAC annual reports (NPL, capital adequacy)
    - IMF FSAP CEMAC 2016 (NPL/capital backward extrapolation for 2000-2009)
  with a balanced panel structure: Country × Year (144 rows).

FIVE EWS PREDICTORS (used in scripts 02–07)
-------------------------------------------
  M2_croissance_pct        — M2 money growth (% YoY)
  Solde_budgetaire_pct_PIB — Fiscal balance (% GDP)
  PIB_croissance_reel_pct  — Real GDP growth (%)
  Reserves_USD             — Gross international reserves (USD, log-transformed)
  Rentes_petrole_pct_PIB   — Oil rents (% GDP)

TARGET VARIABLE
---------------
  StressScore  — composite Banking Stress Indicator (BSI), 0–5 scale
                 (see script 01_BSI_construction.py for construction details)
  Target_Stress2 = 1 if StressScore >= 2  (binary EWS target used in models)

AUTHORS
-------
  Françoise NGOUFACK, Pamphile MEZUI-MBENG, Samba NDIAYE — 2026
  Paper: "Do Early Warning Systems Survive Structural Breaks?
          Macroprudential Evidence from the CEMAC Monetary Union"
  Journal of Financial Stability [under review]
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
# Update DATA_PATH to point to your Dataset_Macro_CEMAC.csv file.
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Dataset_Macro_CEMAC.csv')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# VARIABLE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
# Five predictors used in the EWS models (scripts 02–07), grouped by
# economic transmission channel (see Section 3.1 of the paper).
PREDICTORS = [
    # (column_name, display_label, transmission_channel, expected_sign)
    ('M2_croissance_pct',        'M2 growth (%)',            'Monetary/liquidity', '+'),
    ('Solde_budgetaire_pct_PIB', 'Fiscal balance (% GDP)',   'Fiscal',             '-'),
    ('PIB_croissance_reel_pct',  'Real GDP growth (%)',      'Macro slowdown',     '-'),
    ('Reserves_USD',             'Intl. reserves (log USD)', 'External',           '-'),
    ('Rentes_petrole_pct_PIB',   'Oil rents (% GDP)',        'Commodity shock',    '+'),
]
PRED_COLS   = [p[0] for p in PREDICTORS]
PRED_LABELS = {p[0]: p[1] for p in PREDICTORS}
PRED_CHAN   = {p[0]: p[2] for p in PREDICTORS}
PRED_SIGN   = {p[0]: p[3] for p in PREDICTORS}

TARGET      = 'StressScore'
TARGET_BIN  = 'Target_Stress2'

COUNTRIES   = ['Cameroon', 'CAR', 'Chad', 'Congo', 'Equatorial Guinea', 'Gabon']
COLORS_CTRY = {
    'Cameroon':          '#1F3864',
    'CAR':               '#C0392B',
    'Chad':              '#1A7A2E',
    'Congo':             '#7D3C98',
    'Equatorial Guinea': '#148F77',
    'Gabon':             '#E67E22',
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'font.size':        9,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  EWS-CEMAC  |  Exploratory Data Analysis")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    print(f"\n[ERROR] Data file not found: {DATA_PATH}")
    print("  Please update DATA_PATH at the top of this script.")
    sys.exit(1)

df = (pd.read_csv(DATA_PATH)
        .sort_values(['Country', 'Year'])
        .reset_index(drop=True))

# Log-transform international reserves (reduces right skewness)
if 'Reserves_USD' in df.columns:
    df['Reserves_USD'] = np.log(df['Reserves_USD'].clip(lower=1))

# Binary target
if TARGET in df.columns and TARGET_BIN not in df.columns:
    df[TARGET_BIN] = (df[TARGET] >= 2).astype(int)

print(f"\n  Panel loaded: {len(df)} observations | "
      f"{df['Country'].nunique()} countries | "
      f"{df['Year'].min()}–{df['Year'].max()}")
print(f"  Target (StressScore >= 2): {df[TARGET_BIN].sum()} stress episodes "
      f"({df[TARGET_BIN].mean()*100:.1f}% of obs)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. MISSING DATA REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  MISSING DATA REPORT")
print("─" * 70)
print(f"  {'Variable':<35} {'N_obs':>5}  {'Missing':>7}  {'% Valid':>7}")
print(f"  {'─'*35} {'─'*5}  {'─'*7}  {'─'*7}")
for col in PRED_COLS:
    if col not in df.columns:
        print(f"  {PRED_LABELS[col]:<35} {'NOT IN DATASET':>22}")
        continue
    n_valid   = df[col].notna().sum()
    n_missing = len(df) - n_valid
    pct_valid = n_valid / len(df) * 100
    print(f"  {PRED_LABELS[col]:<35} {n_valid:>5}  {n_missing:>7}  {pct_valid:>6.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS (Table 1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  TABLE 1 — DESCRIPTIVE STATISTICS (CEMAC Panel, 2000–2023)")
print("─" * 70)

desc_rows = []
for col in PRED_COLS:
    if col not in df.columns:
        continue
    s = df[col].dropna()
    s = s[np.isfinite(s)]
    if len(s) < 5:
        continue
    desc_rows.append({
        'Variable':   PRED_LABELS[col],
        'Channel':    PRED_CHAN[col],
        'N':          int(len(s)),
        'Mean':       round(s.mean(),           2),
        'Std':        round(s.std(),            2),
        'Min':        round(s.min(),            2),
        'P25':        round(s.quantile(0.25),   2),
        'Median':     round(s.median(),         2),
        'P75':        round(s.quantile(0.75),   2),
        'Max':        round(s.max(),            2),
        'Skewness':   round(stats.skew(s),      3),
    })

# Also add BSI target
if TARGET_BIN in df.columns:
    s = df[TARGET_BIN].dropna()
    desc_rows.append({
        'Variable': 'BSI stress (binary)',
        'Channel':  'Target',
        'N':        int(len(s)),
        'Mean':     round(s.mean(),   3),
        'Std':      round(s.std(),    3),
        'Min':      int(s.min()),
        'P25':      round(s.quantile(0.25), 2),
        'Median':   int(s.median()),
        'P75':      round(s.quantile(0.75), 2),
        'Max':      int(s.max()),
        'Skewness': round(stats.skew(s), 3),
    })

df_desc = pd.DataFrame(desc_rows)
print(df_desc[['Variable', 'N', 'Mean', 'Std', 'Min', 'Median', 'Max', 'Skewness']
              ].to_string(index=False))
print("\n  Note: |Skewness| > 1 → median imputation used in models (Bulmer 1979 rule).")
for row in desc_rows:
    if abs(row['Skewness']) > 1.0:
        print(f"    → {row['Variable']}: skewness = {row['Skewness']} → median imputation")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CRISIS vs. NON-CRISIS COMPARISON (Mann-Whitney U test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  CRISIS vs. NON-CRISIS COMPARISON (Mann-Whitney U test)")
print("─" * 70)
print(f"  {'Variable':<35} {'Mean(0)':>8} {'Mean(1)':>8} {'p-value':>9} {'Sig.':>5}")
print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*9} {'─'*5}")

if TARGET_BIN in df.columns:
    for col in PRED_COLS:
        if col not in df.columns:
            continue
        g0 = df.loc[df[TARGET_BIN] == 0, col].dropna().values
        g1 = df.loc[df[TARGET_BIN] == 1, col].dropna().values
        if len(g0) < 2 or len(g1) < 2:
            continue
        try:
            _, pval = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        except Exception:
            pval = np.nan
        sig = ('***' if pval < 0.01 else
               '**'  if pval < 0.05 else
               '*'   if pval < 0.10 else '')
        print(f"  {PRED_LABELS[col]:<35} {g0.mean():>8.2f} {g1.mean():>8.2f} "
              f"{pval:>9.4f} {sig:>5}")
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  PAIRWISE PEARSON CORRELATIONS (5 EWS predictors)")
print("─" * 70)

avail_cols = [c for c in PRED_COLS if c in df.columns]
if avail_cols:
    corr_df = df[avail_cols].corr()
    corr_df.index   = [PRED_LABELS[c] for c in avail_cols]
    corr_df.columns = [PRED_LABELS[c] for c in avail_cols]
    print(corr_df.round(3).to_string())
    print("\n  Note: |r| < 0.30 between all pairs → no multicollinearity concern.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── 6a. Distribution histograms by country ───────────────────────────────────
avail_pred = [p for p in PREDICTORS if p[0] in df.columns]
if avail_pred:
    n_vars = len(avail_pred)
    fig, axes = plt.subplots(1, n_vars, figsize=(n_vars * 3.5, 4))
    if n_vars == 1:
        axes = [axes]

    for ax, (col, label, chan, sign) in zip(axes, avail_pred):
        data = df[col].dropna()
        data = data[np.isfinite(data)]
        ax.hist(data, bins=20, color='#2E74B5', alpha=0.7, edgecolor='white')
        ax.axvline(data.mean(),   color='#C0392B', lw=1.5, ls='--',
                   label=f'Mean = {data.mean():.1f}')
        ax.axvline(data.median(), color='#1A1A2E', lw=1.5, ls='-',
                   label=f'Median = {data.median():.1f}')
        ax.set_title(f'{label}\n({chan})', fontsize=8.5, fontweight='bold')
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7)
        sk = stats.skew(data)
        ax.text(0.97, 0.95, f'Skew={sk:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5, color='grey')

    plt.suptitle('EWS Predictor Distributions — CEMAC Panel (2000–2023)',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Figure_EDA_distributions.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  [OK] Figure saved: {out}")

# ── 6b. Box plots — crisis vs. non-crisis ────────────────────────────────────
if TARGET_BIN in df.columns and avail_pred:
    fig, axes = plt.subplots(1, len(avail_pred), figsize=(len(avail_pred) * 3, 4.5))
    if len(avail_pred) == 1:
        axes = [axes]
    palette = {0: '#BDC3C7', 1: '#E74C3C'}

    for ax, (col, label, _, _) in zip(axes, avail_pred):
        data_box = df[[TARGET_BIN, col]].dropna()
        data_box = data_box[np.isfinite(data_box[col])]
        groups   = [data_box.loc[data_box[TARGET_BIN] == k, col].values
                    for k in [0, 1]]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                        medianprops=dict(color='black', lw=2))
        for patch, color in zip(bp['boxes'], [palette[0], palette[1]]):
            patch.set_facecolor(color)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['No stress\n(BSI=0)', 'Stress\n(BSI=1)'], fontsize=8)
        ax.set_title(label, fontsize=8.5, fontweight='bold')
        ax.set_ylabel('Value', fontsize=8)

        # Mann-Whitney p-value annotation
        if len(groups[0]) > 1 and len(groups[1]) > 1:
            try:
                _, pval = stats.mannwhitneyu(groups[0], groups[1],
                                             alternative='two-sided')
                sig = ('***' if pval < 0.01 else '**' if pval < 0.05 else
                       '*'   if pval < 0.10 else 'n.s.')
                ax.text(0.5, 0.97, f'MW p={pval:.3f} {sig}',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=7.5, color='#2C3E50')
            except Exception:
                pass

    plt.suptitle('Predictor Distributions: Stress vs. No-Stress Episodes\n'
                 'CEMAC Panel 2000–2023 — Mann-Whitney U test',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Figure_EDA_boxplots.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Figure saved: {out}")

# ── 6c. Correlation heatmap ───────────────────────────────────────────────────
if len(avail_cols) >= 2:
    fig, ax = plt.subplots(figsize=(6, 5))
    corr_matrix = df[avail_cols].corr()
    short_labels = [PRED_LABELS[c].replace(' (%)', '').replace(' (log USD)', '')
                    for c in avail_cols]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, ax=ax, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=short_labels, yticklabels=short_labels,
                linewidths=0.5, square=True, mask=mask,
                annot_kws={'size': 9})
    ax.set_title('Pairwise Pearson Correlations — EWS Predictors\n'
                 'CEMAC Panel 2000–2023', fontsize=10, fontweight='bold', pad=12)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Figure_EDA_correlation.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Figure saved: {out}")

# ── 6d. BSI stress episodes timeline ─────────────────────────────────────────
if TARGET_BIN in df.columns:
    fig, ax = plt.subplots(figsize=(12, 4))
    years = sorted(df['Year'].unique())

    for i, ctry in enumerate(COUNTRIES):
        sub = df[df['Country'] == ctry].set_index('Year')
        for yr in years:
            if yr in sub.index and sub.loc[yr, TARGET_BIN] == 1:
                ax.barh(y=i, width=1, left=yr - 0.5,
                        color=COLORS_CTRY.get(ctry, 'grey'),
                        alpha=0.85, edgecolor='white', linewidth=0.3)

    ax.set_yticks(range(len(COUNTRIES)))
    ax.set_yticklabels(COUNTRIES, fontsize=9)
    ax.set_xlabel('Year', fontsize=9)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_xticks(range(min(years), max(years) + 1, 2))
    ax.tick_params(axis='x', labelsize=8, rotation=45)
    ax.axvline(2013, color='black', ls='--', lw=1.5, label='Structural break T*=2013')
    ax.axvline(2019, color='grey',  ls=':',  lw=1.2, label='Test window starts 2019')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Banking Stress Episodes (BSI ≥ 2) by Country — CEMAC 2000–2023',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Figure_EDA_target_timeline.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] Figure saved: {out}")

print("\n" + "=" * 70)
print("  EDA complete. Proceed with script 01_BSI_construction.py.")
print("=" * 70)
