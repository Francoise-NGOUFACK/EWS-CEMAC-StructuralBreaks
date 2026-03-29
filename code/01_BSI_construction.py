"""
01_BSI_construction.py
======================
Banking Stress Indicator (BSI) — Threshold Calibration for the CEMAC EWS

PURPOSE
-------
This script constructs the binary target variable (BSI) used in the Early
Warning System (EWS) for the CEMAC banking sector. Each of five component
indicators is assigned an optimal threshold by maximising the Youden Index
(J) from a univariate logistic ROC analysis.  A country-year is classified
as "stressed" (BSI = 1) when at least two of the five component signals fire
simultaneously (≥ 2 criteria rule, consistent with the composite index
approach advocated by Demirgüç-Kunt & Detragiache, 2002).

FIVE COMPONENTS
---------------
  C1  NPL ratio (% gross loans)          — threshold: > 12 %   (above)
  C2  Capital-to-assets ratio (%)        — threshold: < 5 %    (below)
  C3  Private credit growth (% YoY)      — threshold: < 2 %    (below)
  C4  Credit/GDP change (pp)             — threshold: < 0 pp   (below)
  C5  Liquidity support / restructuring  — binary 0/1          (document-coded)

YOUDEN INDEX
------------
For a given threshold τ, the Youden Index is:

    J(τ) = Sensitivity(τ) + Specificity(τ) − 1
         = TPR(τ) − FPR(τ)

The optimal threshold τ* maximises J(τ), balancing the cost of missed crises
(false negatives) against the cost of false alarms (false positives).
Reference: Youden, W.J. (1950). Cancer, 3(1), 32–35.

DATA SOURCES
------------
  World Bank WDI  — https://databank.worldbank.org
  IMF WEO         — https://www.imf.org/en/Publications/WEO
  BEAC / COBAC Annual Reports (NPL, capital adequacy, interventions)
  Panel: 6 CEMAC countries × 2000–2023 (N = 144 country-years)

AUTHORS
-------
  Françoise NGOUFACK, Pamphile MEZUI-MBENG, Samba NDIAYE — 2026
  Paper: "Do Early Warning Systems Survive Structural Breaks?
          Macroprudential Evidence from the CEMAC Monetary Union"
  Journal of Financial Stability [under review]
"""

# %% [markdown]
# # Calibration des Seuils pour la Construction de Target_Severe
# ## Mémoire MDSIA — Système d'Alerte Précoce Bancaire CEMAC
# ### Méthode : Maximisation de l'Indice de Youden sur logit univarié
#
# **Indicateurs testés :**
# 1. NPL (% prêts bruts)
# 2. Ratio Capital / Actifs (%)
# 3. Dette publique (% PIB)
# 4. Croissance du crédit au secteur privé (%)
# 5. Variation crédit / PIB (points de %)
# 6. Intervention de soutien en liquidité / Restructuration bancaire (binaire)

# %% [markdown]
# ## 0. Importations et Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Chemins
PATH_DATA  = "C:/Users/fngou/Desktop/Données_Mémoire ML/"
PATH_BM    = PATH_DATA + "donneeBM.xlsx"
PATH_IMF   = PATH_DATA + "donneeIMF_tauxEchange.csv"
PATH_TGT   = PATH_DATA + "Target_CEMAC.csv"
PATH_OUT   = PATH_DATA + "Calibration_Seuils_Complet.csv"

# Style graphique
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11
})
sns.set_palette("Set2")

COUNTRIES = ['Cameroon', 'CAR', 'Chad', 'Congo', 'Equatorial Guinea', 'Gabon']
YEARS     = list(range(2000, 2024))

print("Librairies chargées avec succès.")

# %% [markdown]
# ## 1. Chargement et Restructuration des Données Banque Mondiale

# %%
bm_raw = pd.read_excel(PATH_BM, sheet_name='Data')

# Colonnes années disponibles
year_cols = {col: int(col[:4]) for col in bm_raw.columns if col[:4].isdigit()}

# Mise en format long
rows = []
for _, row in bm_raw.iterrows():
    for col, year in year_cols.items():
        if 2000 <= year <= 2023:
            val = row[col]
            if str(val) not in ('..', 'nan', ''):
                try:
                    rows.append({
                        'Country': row['Country Name'],
                        'Year':    year,
                        'Series':  row['Series Name'],
                        'Value':   float(val)
                    })
                except:
                    pass

df_long = pd.DataFrame(rows)

# Harmonisation des noms de pays
country_map = {
    'Cameroon':                  'Cameroon',
    'Central African Republic':  'CAR',
    'Chad':                      'Chad',
    'Congo, Rep.':               'Congo',
    'Equatorial Guinea':         'Equatorial Guinea',
    'Gabon':                     'Gabon',
}
df_long['Country'] = df_long['Country'].map(country_map)
df_long = df_long[df_long['Country'].notna()]

# Pivot large
df = df_long.pivot_table(
    index=['Country', 'Year'],
    columns='Series',
    values='Value',
    aggfunc='first'
).reset_index()
df.columns.name = None

print(f"Données BM : {df.shape[0]} obs × {df.shape[1]} colonnes")
print(f"Pays : {sorted(df['Country'].unique())}")
print(f"Années : {df['Year'].min()} – {df['Year'].max()}")
print(f"\nIndicateurs disponibles :")
for col in df.columns[2:]:
    n = df[col].notna().sum()
    print(f"  {col[:65]:65s}: {n:3d}/144 obs")

# %% [markdown]
# ## 2. Calcul des Indicateurs Dérivés

# %%
df = df.sort_values(['Country', 'Year']).reset_index(drop=True)

credit_col = 'Domestic credit to private sector by banks (% of GDP)'

# Croissance du crédit (variation % année/année)
df['Credit_Growth_pct'] = (
    df.groupby('Country')[credit_col]
      .pct_change() * 100
)

# Variation absolue en points de % du PIB
df['Credit_Change_pp'] = (
    df.groupby('Country')[credit_col]
      .diff()
)

print("Variables dérivées calculées :")
print(f"  Credit_Growth_pct  : {df['Credit_Growth_pct'].notna().sum()} obs valides")
print(f"  Credit_Change_pp   : {df['Credit_Change_pp'].notna().sum()} obs valides")

# %% [markdown]
# ## 3. Construction de la Variable d'Intervention (Binaire)
#
# Codage manuel basé sur les 7 rapports lus (FMI 2016/2018/2019/2024/2025,
# COBAC 2024, RSF-AC 2024).
#
# **Critère de codage :** Intervention = 1 si AU MOINS UNE des conditions
# suivantes est documentée pour ce pays-année :
# - Refinancement massif BEAC (facilité de prêt marginal > seuil normal)
# - Plan formel de restructuration COBAC en vigueur
# - Recapitalisation forcée par injonction prudentielle
# - Administration provisoire ou liquidation bancaire
# - Garantie ou recapitalisation par l'État

# %%
# ----------------------------------------------------------------
# Tableau des interventions documentées (source : rapports lus)
# ----------------------------------------------------------------
# Format : (Pays, Année_début, Année_fin, Type, Source)
INTERVENTIONS_DOC = [
    # ── CEMAC-wide ─────────────────────────────────────────────────────
    # Choc pétrolier 2014-2016 → refinancement BEAC extensif à partir de 2016
    # FMI 2018 : "soutien en liquidité de la BEAC a fortement augmenté"
    # FMI 2019 : 15/52 banques en note 4-5 COBAC, plans de redressement
    # FMI 2025 : facilité de prêt marginale FCFA 943 Mds (nov. 2024)

    # ── Cameroun ───────────────────────────────────────────────────────
    ('Cameroon', 2016, 2017, 'Refinancement BEAC massif + plans redressement COBAC', 'FMI 2018'),
    ('Cameroon', 2020, 2023, 'Refinancement BEAC COVID + plans redressement COBAC', 'FMI 2024/2025'),

    # ── RCA ────────────────────────────────────────────────────────────
    # FMI 2016 : effondrement bancaire suite crise politique 2013-2014
    # FMI 2019 : 4 banques en note critique, plans de redressement formels
    ('CAR',      2013, 2016, 'Effondrement bancaire post-crise politique, soutien CEMAC', 'FMI 2016'),
    ('CAR',      2018, 2023, 'Plans redressement COBAC + refinancement BEAC', 'FMI 2019/2024'),

    # ── Tchad ──────────────────────────────────────────────────────────
    # FMI 2025 : 74.7% du déficit en fonds propres régional
    # COBAC 2024 : fonds propres négatifs en 2023, injonctions de recapitalisation
    ('Chad',     2016, 2023, 'Injonctions recapitalisation COBAC + refinancement BEAC massif', 'COBAC 2024 / FMI 2025'),

    # ── Congo ──────────────────────────────────────────────────────────
    # FMI 2018 : violations significatives ratios prudentiels (liquidité)
    # RSF-AC 2024 : seul pays avec contraction du crédit (-2.9%)
    # FMI 2025 : Congo "en détresse de dette"
    ('Congo',    2016, 2023, 'Violations prudentielles, restructuration, refinancement BEAC', 'FMI 2018 / RSF-AC 2024'),

    # ── Guinée Équatoriale ─────────────────────────────────────────────
    # FMI 2018 : violations importantes ratios liquidité en 2016
    # COBAC 2024 : fonds propres négatifs 3 années consécutives
    ('Equatorial Guinea', 2016, 2017, 'Violations liquidité + plans redressement COBAC', 'FMI 2018'),
    ('Equatorial Guinea', 2018, 2023, 'Fonds propres négatifs, injonctions recapitalisation COBAC', 'COBAC 2024 / RSF-AC 2024'),

    # ── Gabon ──────────────────────────────────────────────────────────
    # FMI 2016 : NPL Gabon déjà à 19.7-20.1% en 2012-2013 (le plus élevé CEMAC)
    # RSF-AC 2024 : pire dégradation NPL (+31.4%)
    # FMI 2025 : Gabon sous SMP (post coup d'État août 2023)
    ('Gabon',    2015, 2023, 'Restructuration bancaire + refinancement BEAC + SMP FMI 2024', 'FMI 2016/2025 / RSF-AC 2024'),
]

# Construire la variable binaire pays × année
intervention_rows = []
for country in COUNTRIES:
    for year in YEARS:
        val = 0
        sources = []
        for (c, y1, y2, typ, src) in INTERVENTIONS_DOC:
            if c == country and y1 <= year <= y2:
                val = 1
                sources.append(src)
        intervention_rows.append({
            'Country':      country,
            'Year':         year,
            'Intervention': val,
            'Source_doc':   ' | '.join(sources) if sources else ''
        })

df_interv = pd.DataFrame(intervention_rows)

print("Variable Intervention (binaire) — Répartition :")
print(df_interv.groupby('Country')['Intervention'].agg(['sum','count'])
        .rename(columns={'sum':'Nb_intervention','count':'Nb_obs'}))
print(f"\nTaux d'événement global : {df_interv['Intervention'].mean()*100:.1f}%")

# %% [markdown]
# ## 4. Fusion du Dataset Final

# %%
# Charger Target_Severe
target = (pd.read_csv(PATH_TGT)
            .query("2000 <= Year <= 2023")[['Country','Year','Target']]
            .rename(columns={'Target': 'Target_Severe'}))

# Fusion BM + Intervention + Target
df_final = (df
    .merge(df_interv[['Country','Year','Intervention']], on=['Country','Year'], how='left')
    .merge(target, on=['Country','Year'], how='inner')
)

print(f"Dataset final : {df_final.shape[0]} obs × {df_final.shape[1]} colonnes")
print(f"Target_Severe — taux d'événement : {df_final['Target_Severe'].mean()*100:.1f}%")
print(f"Intervention  — taux d'événement : {df_final['Intervention'].mean()*100:.1f}%")

# %% [markdown]
# ## 5. Fonctions d'Évaluation

# %%
def evaluate_threshold(y_true, signal):
    """
    Compute Sensitivity, Specificity, Youden Index, and AUC-ROC
    for a binary signal (0/1) against y_true.

    Youden Index: J = TPR + TNR − 1 = Sensitivity + Specificity − 1
      J = 0  → no discriminatory power (random classifier)
      J = 1  → perfect classifier
    Minimum acceptable for EWS: J > 0.20 (Kaminsky et al., 1998).

    Returns None if fewer than 10 observations or if y_true is constant
    (AUC undefined when only one class is present in the sample).
    """
    y  = np.array(y_true, dtype=float)
    s  = np.array(signal, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(s)
    y, s = y[mask], s[mask]

    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return None

    TP = ((s == 1) & (y == 1)).sum()
    FN = ((s == 0) & (y == 1)).sum()
    TN = ((s == 0) & (y == 0)).sum()
    FP = ((s == 1) & (y == 0)).sum()

    sens   = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec   = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    youden = sens + spec - 1.0

    try:
        auc = roc_auc_score(y, s)
    except Exception:
        auc = np.nan

    return dict(N_obs=int(mask.sum()), N_crises=int(y.sum()),
                N_signal=int(s.sum()),
                TP=int(TP), FP=int(FP), TN=int(TN), FN=int(FN),
                Sensibilite=round(sens,   4),
                Specificite=round(spec,   4),
                Youden     =round(youden, 4),
                AUC        =round(auc,    4) if not np.isnan(auc) else np.nan)


def full_roc_auc(y_true, x_values, direction='above'):
    """AUC globale de l'indicateur continu via courbe ROC complète."""
    mask = pd.notna(y_true) & pd.notna(x_values)
    y = y_true[mask].values.astype(float)
    x = x_values[mask].values.astype(float)
    if len(y) < 10 or y.sum() == 0:
        return np.nan
    score = -x if direction == 'below' else x
    try:
        return round(roc_auc_score(y, score), 4)
    except Exception:
        return np.nan

print("Fonctions d'évaluation définies.")

# %% [markdown]
# ## 6. Définition des Indicateurs et Seuils à Tester

# %%
# Format : (label_affichage, colonne_df, direction, liste_seuils)
#   direction 'above' → signal = 1 si valeur > seuil  (NPL, dette)
#   direction 'below' → signal = 1 si valeur < seuil  (capital, crédit)
#   direction 'binary'→ indicateur déjà binaire

INDICATORS = [
    (
        'NPL (% prêts bruts)',
        'Bank nonperforming loans to total gross loans (%)',
        'above',
        [8, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25]
    ),
    (
        'Capital / Actifs (%)',
        'Bank capital to assets ratio (%)',
        'below',
        [3, 4, 5, 6, 7, 8, 9, 10, 12]
    ),
    (
        'Dette publique (% PIB)',
        'Central government debt, total (% of GDP)',
        'above',
        [30, 40, 50, 60, 70, 80, 90, 100]
    ),
    (
        'Croissance crédit privé (%)',
        'Credit_Growth_pct',
        'below',
        [-20, -15, -10, -7, -5, -3, 0, 2, 5]
    ),
    (
        'Variation crédit / PIB (pp)',
        'Credit_Change_pp',
        'below',
        [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
    ),
    (
        'Intervention soutien / Restructuration (binaire)',
        'Intervention',
        'binary',
        [0.5]      # seuil unique pour une variable déjà binaire
    ),
]

print("Indicateurs et seuils configurés.")

# %% [markdown]
# ## 7. Analyse par Indicateur — Tables et Graphiques

# %%
all_results = []

for (label, col, direction, thresholds) in INDICATORS:

    # ── Vérification disponibilité ─────────────────────────────────────
    if col not in df_final.columns:
        print(f"\n[ABSENT] {label} — colonne '{col}' introuvable.")
        continue

    n_valid = df_final[col].notna().sum()
    auc_global = full_roc_auc(df_final['Target_Severe'], df_final[col], direction) \
                 if direction != 'binary' else np.nan

    stats = df_final[col].dropna().describe()
    print(f"\n{'═'*68}")
    print(f"  {label}")
    print(f"  Obs. valides : {n_valid}/144  |  AUC globale : {auc_global}")
    print(f"  Min={stats['min']:.2f}  Moy={stats['mean']:.2f}  "
          f"Méd={stats['50%']:.2f}  Max={stats['max']:.2f}")
    print(f"{'─'*68}")

    # ── Boucle seuils ─────────────────────────────────────────────────
    records = []
    for t in thresholds:
        if direction == 'above':
            sig = np.where(df_final[col].isna(), np.nan,
                           (df_final[col] > t).astype(float))
        elif direction == 'below':
            sig = np.where(df_final[col].isna(), np.nan,
                           (df_final[col] < t).astype(float))
        else:  # binary
            sig = df_final[col].values.astype(float)

        res = evaluate_threshold(df_final['Target_Severe'], sig)
        if res is None:
            continue
        records.append({'Seuil': t, **res,
                        'Indicateur': label,
                        'Direction': direction,
                        'AUC_globale': auc_global})

    if not records:
        print("  Pas assez d'observations pour calculer les métriques.")
        continue

    df_rec = pd.DataFrame(records)
    best_idx = df_rec['Youden'].idxmax()

    # Affichage tableau console
    header = (f"  {'Seuil':>7} | {'N_obs':>5} | {'Sig':>4} | "
              f"{'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} | "
              f"{'Sens.':>6} {'Spec.':>6} {'Youden':>7} {'AUC':>6}")
    print(header)
    print(f"  {'─'*65}")
    for _, r in df_rec.iterrows():
        marker = ' ◀ OPTIMAL' if r.name == best_idx else ''
        auc_str = f"{r['AUC']:.3f}" if not np.isnan(r['AUC']) else '  —  '
        print(f"  {r['Seuil']:>7.1f} | {r['N_obs']:>5} | {r['N_signal']:>4} | "
              f"{r['TP']:>4} {r['FP']:>4} {r['TN']:>4} {r['FN']:>4} | "
              f"{r['Sensibilite']:>6.3f} {r['Specificite']:>6.3f} "
              f"{r['Youden']:>7.3f} {auc_str:>6}{marker}")

    opt = df_rec.loc[best_idx]
    print(f"\n  ▶ Seuil optimal : {opt['Seuil']}  "
          f"(Youden={opt['Youden']:.3f} | "
          f"Sens={opt['Sensibilite']:.3f} | "
          f"Spec={opt['Specificite']:.3f})")

    all_results.extend(df_rec.to_dict('records'))

    # ── Graphique : Courbe ROC + Youden ───────────────────────────────
    if direction != 'binary' and n_valid >= 20:
        x_vals = df_final[col].values.astype(float)
        y_vals = df_final['Target_Severe'].values.astype(float)
        mask   = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        score  = (-x_vals[mask] if direction == 'below' else x_vals[mask])

        fpr, tpr, _ = roc_curve(y_vals[mask], score)
        youden_vals  = df_rec['Youden'].values
        seuil_vals   = df_rec['Seuil'].values

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # — ROC curve —
        ax = axes[0]
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_global:.3f}')
        ax.plot([0,1],[0,1],'--', color='grey', lw=1)
        ax.fill_between(fpr, tpr, alpha=0.08)
        ax.set_xlabel('1 − Spécificité  (Taux faux positifs)')
        ax.set_ylabel('Sensibilité  (Taux vrais positifs)')
        ax.set_title(f'Courbe ROC — {label}')
        ax.legend(loc='lower right')
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)

        # — Youden Index vs seuil —
        ax = axes[1]
        ax.plot(seuil_vals, youden_vals, 'o-', lw=2, ms=7)
        best_t  = opt['Seuil']
        best_y  = opt['Youden']
        ax.axvline(best_t, color='red', ls='--', lw=1.2,
                   label=f'Optimal = {best_t} → J={best_y:.3f}')
        ax.scatter([best_t], [best_y], color='red', zorder=5, s=90)
        sens_vals = df_rec['Sensibilite'].values
        spec_vals = df_rec['Specificite'].values
        ax.plot(seuil_vals, sens_vals, 's--', lw=1, ms=5,
                alpha=0.7, label='Sensibilité')
        ax.plot(seuil_vals, spec_vals, '^--', lw=1, ms=5,
                alpha=0.7, label='Spécificité')
        ax.set_xlabel('Seuil')
        ax.set_ylabel('Valeur')
        ax.set_title(f'Youden Index par seuil — {label}')
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        plt.suptitle(label, fontsize=12, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()

    # Graphique simplifié pour variable binaire
    elif direction == 'binary':
        opt_r = df_rec.iloc[0]
        fig, ax = plt.subplots(figsize=(6,4))
        metrics = ['Sensibilité','Spécificité','Youden']
        values  = [opt_r['Sensibilite'], opt_r['Specificite'], opt_r['Youden']]
        colors  = ['#2ecc71','#3498db','#e74c3c']
        bars = ax.barh(metrics, values, color=colors, height=0.5)
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=11)
        ax.set_xlim(0, 1.15)
        ax.set_title(f'Performance — {label}', fontweight='bold')
        ax.axvline(0.5, color='grey', ls='--', lw=1)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 8. Tableau Récapitulatif — Seuils Optimaux

# %%
df_all = pd.DataFrame(all_results)

# Un seul seuil optimal par indicateur (max Youden)
summary_rows = []
for indic, grp in df_all.groupby('Indicateur'):
    best = grp.loc[grp['Youden'].idxmax()]
    direction_label = {'above': '>', 'below': '<', 'binary': 'binaire'}[best['Direction']]
    summary_rows.append({
        'Indicateur':    indic,
        'Seuil optimal': f"{direction_label} {best['Seuil']:.1f}" if best['Direction'] != 'binary' else 'Oui/Non',
        'N obs':         int(best['N_obs']),
        'AUC globale':   best['AUC_globale'] if not np.isnan(best['AUC_globale']) else '—',
        'Sensibilité':   f"{best['Sensibilite']:.3f}",
        'Spécificité':   f"{best['Specificite']:.3f}",
        'Youden':        f"{best['Youden']:.3f}",
        'Qualité':       ('★★★' if best['AUC_globale'] > 0.65
                          else '★★' if best['AUC_globale'] > 0.55
                          else '★' if not np.isnan(best['AUC_globale'])
                          else '(binaire)')
    })

df_summary = pd.DataFrame(summary_rows)
# Trier par Youden décroissant
df_summary = df_summary.sort_values('Youden', ascending=False).reset_index(drop=True)

print("\n" + "═"*85)
print("  TABLEAU RÉCAPITULATIF — SEUILS OPTIMAUX PAR INDICATEUR")
print("  (Méthode : Maximisation de l'Indice de Youden)")
print("═"*85)
print(df_summary.to_string(index=False))

# %% [markdown]
# ## 9. Graphique Comparatif — Tous les Indicateurs

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

indic_labels = df_summary['Indicateur'].tolist()
sens_vals    = [float(x) for x in df_summary['Sensibilité']]
spec_vals    = [float(x) for x in df_summary['Spécificité']]
you_vals     = [float(x) for x in df_summary['Youden']]

x = np.arange(len(indic_labels))
short_labels = [
    l.replace('(% prêts bruts)','').replace('(%)','').replace('(pp)','').strip()
     .replace('/ PIB ','').replace('Variation crédit','Crédit Δpp')
     .replace('Croissance crédit privé','Crédit Growth')
     .replace('Intervention soutien / Restructuration (binaire)','Intervention')
    for l in indic_labels
]

# Sensibilité & Spécificité
ax = axes[0]
w = 0.35
ax.bar(x - w/2, sens_vals, w, label='Sensibilité', color='#2ecc71')
ax.bar(x + w/2, spec_vals, w, label='Spécificité', color='#3498db')
ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title('Sensibilité & Spécificité', fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0.5, color='grey', ls='--', lw=1)

# Youden Index
ax = axes[1]
colors_you = ['#e74c3c' if v == max(you_vals) else '#95a5a6' for v in you_vals]
bars = ax.bar(x, you_vals, color=colors_you)
ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, max(you_vals) * 1.25)
for bar, val in zip(bars, you_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_title('Indice de Youden', fontweight='bold')
ax.axhline(0.2, color='grey', ls='--', lw=1, label='Seuil acceptable (0.2)')
ax.legend(fontsize=9)

# AUC globale
ax = axes[2]
auc_vals_plot = []
for v in df_summary['AUC globale']:
    try:
        auc_vals_plot.append(float(v))
    except:
        auc_vals_plot.append(0.5)
colors_auc = ['#e74c3c' if v == max(auc_vals_plot) else '#95a5a6' for v in auc_vals_plot]
bars = ax.bar(x, auc_vals_plot, color=colors_auc)
ax.axhline(0.5,  color='black', ls='-',  lw=1.5, label='Hasard (0.5)')
ax.axhline(0.65, color='orange',ls='--', lw=1,   label='Bon (0.65)')
ax.axhline(0.75, color='green', ls='--', lw=1,   label='Très bon (0.75)')
ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=9)
ax.set_ylim(0, 1.0)
for bar, val in zip(bars, auc_vals_plot):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=9)
ax.set_title('AUC globale (courbe ROC)', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('Comparaison des Indicateurs — Calibration des Seuils (Target_Severe)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Construction de Target_Severe Calibrée
#
# On combine les indicateurs selon la règle **≥ 2 critères simultanés = crise**
# (cohérent avec la méthodologie du Chapitre 3).

# %%
npl_col   = 'Bank nonperforming loans to total gross loans (%)'
cap_col   = 'Bank capital to assets ratio (%)'
cred_col  = 'Credit_Growth_pct'
credp_col = 'Credit_Change_pp'

# Seuils optimaux issus de la calibration
df_final['C1_NPL']         = (df_final[npl_col]   > 12).astype(float)
df_final['C2_Capital']     = (df_final[cap_col]    <  5).astype(float)
df_final['C3_CreditGrowth']= (df_final[cred_col]   <  2).astype(float)
df_final['C4_CreditPP']    = (df_final[credp_col]  <  0).astype(float)
df_final['C5_Intervention'] = df_final['Intervention'].astype(float)

# Remplacer NaN par 0 pour le comptage (critère non applicable = non déclenché)
crit_cols = ['C1_NPL','C2_Capital','C3_CreditGrowth','C4_CreditPP','C5_Intervention']
df_final['N_criteres']     = df_final[crit_cols].fillna(0).sum(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE RULE: a country-year is flagged as "banking stress" when at least
# k out of 5 component indicators fire simultaneously.
#
#   BSI(k) = 1  iff  Σ Ci ≥ k,  i ∈ {C1…C5}
#
# k = 2 (main specification): balances sensitivity and specificity; consistent
#   with composite EWS methodology (Demirgüç-Kunt & Detragiache, 2002;
#   ECB CISS index design).
# k = 3 (robustness check): more conservative, higher specificity.
# ─────────────────────────────────────────────────────────────────────────────
df_final['Target_Severe_Calibre_2'] = (df_final['N_criteres'] >= 2).astype(int)
df_final['Target_Severe_Calibre_3'] = (df_final['N_criteres'] >= 3).astype(int)

# Évaluation des deux règles
print("ÉVALUATION DES RÈGLES DE COMBINAISON")
print("="*55)
for rule_col, rule_label in [
    ('Target_Severe_Calibre_2', 'Règle ≥ 2 critères'),
    ('Target_Severe_Calibre_3', 'Règle ≥ 3 critères'),
]:
    res = evaluate_threshold(df_final['Target_Severe'], df_final[rule_col])
    if res:
        print(f"\n  {rule_label}")
        print(f"    Taux événement  : {df_final[rule_col].mean()*100:.1f}%")
        print(f"    Sensibilité     : {res['Sensibilite']:.3f}")
        print(f"    Spécificité     : {res['Specificite']:.3f}")
        print(f"    Youden          : {res['Youden']:.3f}")
        print(f"    AUC             : {res['AUC']:.3f}")
        print(f"    TP={res['TP']}  FP={res['FP']}  TN={res['TN']}  FN={res['FN']}")

# %% [markdown]
# ## 11. Sauvegarde des Résultats

# %%
# Résultats complets de calibration
df_all.to_csv(PATH_OUT, index=False, encoding='utf-8-sig')
print(f"Résultats calibration sauvegardés : {PATH_OUT}")

# Dataset final avec toutes les variables
cols_save = ['Country','Year','Target_Severe',
             npl_col, cap_col, cred_col, credp_col, 'Intervention',
             'C1_NPL','C2_Capital','C3_CreditGrowth','C4_CreditPP','C5_Intervention',
             'N_criteres','Target_Severe_Calibre_2','Target_Severe_Calibre_3']
cols_save = [c for c in cols_save if c in df_final.columns]

out_final = PATH_DATA + "Dataset_Target_Calibre.csv"
df_final[cols_save].to_csv(out_final, index=False, encoding='utf-8-sig')
print(f"Dataset final sauvegardé           : {out_final}")

print("\nTerminé.")
