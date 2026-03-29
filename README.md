# Do Early Warning Systems Survive Structural Breaks? Macroprudential Evidence from the CEMAC Monetary Union

**Replication code and data for the paper submitted to the *Journal of Financial Stability***

---

## Authors

- **Françoise NGOUFACK** *(Corresponding author)* — Université Omar Bongo / ESMT
  francois.ngoufack.etu@esmt.sn
- **Pamphile MEZUI-MBENG** — Université Omar Bongo
- **Samba NDIAYE** — Université Cheikh Anta Diop / ESMT

---

## Abstract

Banking sector fragility in the Central African Economic and Monetary Community (CEMAC) has intensified since the mid-2010s, reflecting both the end of the oil boom and persistent institutional weaknesses in financial supervision. Drawing on annual data for the six CEMAC member states over 2000–2023, this paper asks how useful macroprudential early warning systems (EWS) can be in such a small, commodity-dependent monetary union, and whether their performance is robust to macro-financial regime shifts. All three classifiers (penalised logistic regression, Random Forest, XGBoost) achieve AUC-ROC above 0.70 on pre-2013 data but their discriminatory power collapses to 0.46–0.54 in the 2019–2023 test window. A Supremum Likelihood Ratio Test identifies 2013 as a statistically significant structural break (p < 0.001), confirmed by Monte Carlo simulations. The main policy lesson is that improving EWS performance requires regime-aware calibration and systematic model governance, not more sophisticated algorithms.

**JEL codes:** G21 · C53 · O55
**Keywords:** Early warning system; Banking fragility; Structural break; CEMAC; Macroprudential calibration; Model governance

---

## Repository Structure

```
EWS-CEMAC-StructuralBreaks/
│
├── code/
│   ├── 00_data_analysis.py               # Exploratory data analysis — descriptive stats, distributions, correlations
│   ├── 01_BSI_construction.py            # Banking Stress Indicator — threshold calibration
│   ├── 02_logit_L2_reference_split.py    # Penalised logistic regression (Logit-L2), 2019 split
│   ├── 03_random_forest.py               # Random Forest classifier, 2019 split
│   ├── 04_xgboost.py                     # XGBoost classifier, 2019 split
│   ├── 05_expanding_window_validation.py # Expanding-window temporal validation (5 cutoffs)
│   ├── 06_structural_stability_tests.py  # SupLRT test + bootstrap coefficient diagnostics
│   └── 07_monte_carlo_benchmark.py       # Monte Carlo benchmark (Variants A and B)
│
├── data/
│   ├── EWS_rolling_auc.csv               # AUC-ROC across expanding-window splits (Table 3)
│   ├── EWS_moniteur_suplrt.csv           # SupLRT scan results (Table 5)
│   ├── EWS_signaux_operationnels.csv     # Operational early-warning signals
│   └── monte_carlo_results.csv           # Monte Carlo AUC distributions (Table 6)
│
├── figures/
│   ├── Figure_1_BSI_distribution.png     # BSI distribution and temporal profile
│   ├── Figure_2_expanding_AUC.png        # AUC-ROC across expanding-window splits
│   ├── Figure_3_ROC_curves.png           # ROC curves on reference test window (2019–2023)
│   ├── Figure_4_SupLRT_profile.png       # SupLRT profile and break-date stability
│   └── Figure_A_bootstrap_stability.png  # Bootstrap coefficient stability (Appendix)
│
├── requirements.txt                      # Python dependencies
└── README.md
```

---

## Data Sources

The macroeconomic panel (6 CEMAC countries, 2000–2023, N = 144 country-years) was assembled from:

| Variable | Source |
|---|---|
| Real GDP growth, M2 growth, fiscal balance/GDP, public debt/GDP | World Bank WDI — https://databank.worldbank.org |
| Oil rents/GDP, international reserves | IMF World Economic Outlook — https://www.imf.org/en/Publications/WEO |
| NPL ratios (% gross loans), bank capital-to-assets ratio — **2010–2023** | World Bank WDI / Global Financial Development Database (observed) |
| NPL ratios, bank capital-to-assets ratio — **2000–2009** | IMF FSAP CEMAC 2016 (backward extrapolation from observed 2010 anchor) |
| Public liquidity support / banking restructuring interventions | BEAC and COBAC Annual Reports and Financial Stability Reports (document-coded binary variable) |

> **Note:** NPL and capital adequacy data for 2010–2023 are taken from the World Bank WDI/GFDD (`Bank nonperforming loans to total gross loans (%)` and `Bank capital to assets ratio (%)`). Values for 2000–2009 are not directly available in public databases for all CEMAC countries and were extrapolated backward from the 2010 observed anchor using the trend documented in the IMF FSAP CEMAC 2016 report. The `data/` folder contains only model output files (AUC scores, SupLRT statistics, Monte Carlo distributions). Raw panel data are not redistributed; please consult the sources above to reconstruct them.

---

## How to Replicate

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the scripts in order

```bash
# Step 0 — Exploratory data analysis (descriptive statistics, distributions, correlations)
python code/00_data_analysis.py

# Step 1 — Construct the Banking Stress Indicator
python code/01_BSI_construction.py

# Step 2 — Estimate the three classifiers (reference split: train 2000–2018, test 2019–2023)
python code/02_logit_L2_reference_split.py
python code/03_random_forest.py
python code/04_xgboost.py

# Step 3 — Expanding-window temporal validation (5 cutoffs: 2015–2019)
python code/05_expanding_window_validation.py

# Step 4 — Structural stability tests (SupLRT + bootstrap coefficient diagnostics)
python code/06_structural_stability_tests.py

# Step 5 — Monte Carlo benchmark (Variants A and B, 500 synthetic panels)
python code/07_monte_carlo_benchmark.py
```

### 3. Software versions used in the paper

| Package | Version used in paper | Current repo version |
|---|---|---|
| Python | 3.10+ | 3.10+ |
| pandas | 2.2.3 | 2.2.3 |
| numpy | 2.1.3 | 2.1.3 |
| scikit-learn | 1.3 | 1.8.0 |
| xgboost | 1.7 | 3.2.0 |
| scipy | 1.17.0 | 1.17.0 |
| statsmodels | 0.14.6 | 0.14.6 |
| matplotlib | 3.9.2 | 3.9.2 |
| seaborn | 0.13.2 | 0.13.2 |
| shap | 0.50.0 | 0.50.0 |

> Results were generated with **scikit-learn v1.3** and **XGBoost v1.7** as stated in Appendix B of the paper. Minor numerical differences may arise with later versions due to changes in default hyperparameters or random seed handling. All random seeds are fixed at `seed = 42`.

---

## Key Results

| Model | AUC-ROC (train 2000–2018, in-sample) | AUC-ROC (test 2019–2023) |
|---|---|---|
| Logit-L2 | 0.728 | 0.488 |
| Random Forest | 0.715 | 0.459 |
| XGBoost | 0.722 | 0.541 |

**Structural break:** SupLRT statistic = 44.4 at T* = 2013 (bootstrap p-value < 0.001).
**Monte Carlo:** 87–95% of synthetic panels with a break at 2013 yield AUC < 0.60, vs. < 0.5% under a stationary DGP.

---

## Citation

If you use this code or data, please cite:

```
NGOUFACK, F., MEZUI-MBENG, P., & NDIAYE, S. (2026). Do Early Warning Systems Survive
Structural Breaks? Macroprudential Evidence from the CEMAC Monetary Union.
Journal of Financial Stability. [Under review]
```

---

## License

This code is released under the **MIT License**. See `LICENSE` for details.
The data files in `data/` are released under **CC BY 4.0**.
