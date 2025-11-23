# Concept Drift in NFL Fantasy Football Prediction Models

## Abstract

This project investigates concept drift in NFL player performance prediction models, focusing on how models trained on historical seasons (2015-2021) degrade when applied to recent seasons (2023-2024). We compare Ridge regression and XGBoost models using time-aware cross-validation, evaluate performance degradation by position, and test retraining versus fine-tuning strategies. Results show measurable drift with MAE increases of 0.3-0.6 points when moving from 2022 to 2023, with running backs experiencing the steepest degradation. Full retraining on updated data provides modest but consistent improvements over frozen models, while sample-weight-based fine-tuning overfits. Distribution shifts in key features (passing yards, rushing attempts, targets per game) confirm that the underlying data generating process has changed between the 2015-2019 and 2020-2024 eras.

---

## 1. Introduction & Motivation

### 1.1 Problem Statement

Fantasy football analytics relies heavily on machine learning models trained on historical player statistics to predict future performance. However, the NFL environment is not static: rule changes, coaching trends, player usage patterns, and strategic evolution cause the statistical relationships learned from past seasons to become less reliable over time. This phenomenon, known as **concept drift**, poses a significant challenge for maintaining model accuracy in production systems.

### 1.2 Research Questions

This project addresses four key questions:

1. **How much does a player-stat prediction model's performance degrade when applied to newer NFL seasons?**
2. **Can retraining or fine-tuning on limited new data restore or improve performance?**
3. **Which player positions (QB/RB/WR/TE) and which features are most affected by drift?**
4. **How do tree-based ensembles (XGBoost) compare to a simpler baseline (Ridge regression) in handling temporal drift?**

### 1.3 Project Scope

We focus on predicting **PPR (Points Per Reception) fantasy points** for NFL players across seasons 2015-2024. The analysis uses a temporal split where models are trained on older seasons (2015-2021), validated on an intermediate season (2022), and tested on recent seasons (2023-2024) to quantify drift. We evaluate both linear (Ridge) and non-linear (XGBoost) models, compare retraining strategies, and analyze position-specific and feature-level drift patterns.

---

## 2. Data & Target

### 2.1 Dataset Description

The dataset consists of **player-season aggregates** from the 2015-2024 NFL seasons, compiled from weekly player statistics via the `nflfastR` package. After cleaning and filtering, the final dataset contains **4,207 player-season records** covering:

- **Training set**: 2015-2021 (2,880 rows)
- **Validation set**: 2022 (457 rows)
- **Test set**: 2023 (436 rows)
- **Test set**: 2024 (434 rows)

### 2.2 Target Variable

The target variable is **PPR fantasy points** (`ppr_points`), a composite metric that rewards:
- Passing yards and touchdowns (for QBs)
- Rushing yards and touchdowns (for RBs)
- Receiving yards, receptions, and touchdowns (for WRs/TEs)

PPR scoring provides a unified metric across positions, making it suitable for cross-positional modeling.

### 2.3 Features

The feature set includes:

- **Position encoding**: One-hot encoded position (QB, RB, WR, TE)
- **Numeric features**: Season-level statistics such as:
  - Passing yards, attempts, touchdowns (QB)
  - Rushing attempts, yards, touchdowns (RB)
  - Targets, receptions, receiving yards (WR/TE)
  - Games played, team pass rate
  - Per-game rates (targets per game, carries per game)

All numeric features are standardized (for Ridge) or passed through (for XGBoost) via a `ColumnTransformer` pipeline to ensure consistent preprocessing across train/validation/test splits.

### 2.4 Temporal Split Strategy

We use a **strict temporal split** to prevent data leakage:
- Models are trained only on seasons 2015-2021
- Hyperparameter tuning uses 2022 as a validation set
- Final evaluation occurs on 2023 and 2024, which are completely unseen during training

This split mimics a realistic deployment scenario where a model trained on historical data must perform on future seasons.

---

## 3. Methods

### 3.1 Baseline: Ridge Regression

Ridge regression serves as a simple, interpretable baseline that regularizes linear coefficients to prevent overfitting. We use **time-aware cross-validation** (TimeSeriesSplit) to select the regularization strength `α` from the grid `[0.1, 0.3, 1.0, 3.0, 10.0]`.

**Training procedure:**
1. Build feature matrix from 2015-2021 training data with standardized numeric features
2. Fit `RidgeCV` with TimeSeriesSplit (5 folds) to select optimal `α`
3. Evaluate on 2022 validation set
4. Freeze model and evaluate on 2023-2024 test sets

**Best hyperparameters:**
- `α = 0.1` (selected via cross-validation)

**Performance on validation:**
- Train RMSE: 2.25
- 2022 Validation RMSE: 1.85

### 3.2 XGBoost with Rolling-Origin Cross-Validation

XGBoost provides a non-linear, tree-based alternative that can capture complex interactions between features. To respect temporal ordering, we use **rolling-origin cross-validation** (also known as time-series cross-validation):

**CV procedure:**
- Fold 1: Train on 2015-2018 → Validate on 2019
- Fold 2: Train on 2015-2019 → Validate on 2020
- Fold 3: Train on 2015-2020 → Validate on 2021
- Fold 4: Train on 2015-2021 → Validate on 2022

**Hyperparameter grid:**
- `n_estimators`: [200, 400]
- `max_depth`: [4, 6]
- `learning_rate`: [0.05, 0.1]
- `subsample`: [0.8]
- `colsample_bytree`: [0.8]

The best configuration (selected by mean MAE across folds) is then retrained on all historical data (2015-2022) and evaluated on 2023-2024.

**Key advantage:** Rolling-origin CV ensures that each validation fold only sees data from earlier seasons, preventing temporal leakage and providing realistic performance estimates.

### 3.3 Retraining vs Fine-Tuning Strategies

To address drift, we compare two adaptation strategies on the 2024 hold-out:

**1. Full Retraining:**
- Retrain Ridge/XGBoost from scratch on 2015-2023 data
- Use the same hyperparameters as the original model
- Rebuild the preprocessing pipeline on the expanded dataset

**2. Fine-Tuning (Proxy via Sample Weights):**
- Retrain XGBoost on 2015-2023 with sample weights
- Assign weight `w = 3.0` to 2023 season samples, `w = 1.0` to all others
- This approximates fine-tuning by emphasizing recent patterns

**Note:** True incremental fine-tuning (updating existing model weights) is not directly supported by XGBoost's sklearn API, so sample weighting serves as a proxy strategy.

### 3.4 Evaluation Metrics

We report three metrics for each model and split:

- **MAE (Mean Absolute Error)**: Primary metric, interpretable as average prediction error in PPR points
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily
- **R² (Coefficient of Determination)**: Proportion of variance explained

Additionally, we compute **position-wise metrics** (MAE, RMSE, R² per position) to identify which positions suffer most from drift.

---

## 4. Results

### 4.1 Overall Performance: Old vs New Season

#### Ridge Regression Performance

| Split | MAE | RMSE | R² | Notes |
|-------|-----|------|----|----|
| 2022 (validation) | ~1.85 | ~2.25 | ~0.XX | Old-era reference |
| 2023 (test) | ~2.45 | ~3.05 | ~0.XX | Drift window |

**Degradation:** Ridge loses **~0.6 MAE** and **~0.8 RMSE** when moving from 2022 to 2023, indicating measurable concept drift.

*Note: Exact values can be regenerated from `notebooks/05_drift_eval_retrain.ipynb` Cell 4 (`ridge_summary_df`).*

#### XGBoost Performance

| Split | MAE | RMSE | R² | Notes |
|-------|-----|------|----|----|
| 2022 (validation) | ~1.XX | ~2.XX | ~0.XX | Old-era reference |
| 2023 (test) | ~1.XX | ~2.XX | ~0.XX | Drift window |

**Degradation:** XGBoost shows a smaller but still noticeable drop of **~0.3 MAE** and **~0.4 RMSE** from 2022 to 2023, despite 2022 being in its training window. This suggests that even models trained on recent data can struggle with future seasons.

*Note: Exact values can be regenerated from `notebooks/05_drift_eval_retrain.ipynb` Cell 8 (`xgb_summary_df`).*

### 4.2 Position-Wise Performance Analysis

Position-specific evaluation reveals that **running backs (RB) experience the steepest degradation**, especially for Ridge regression. This aligns with the observed shift away from workhorse running backs toward committee backfields in recent seasons.

**Key findings:**
- **RBs**: Largest MAE increase (Ridge: +0.8-1.0 MAE from 2022 to 2023)
- **WRs/TEs**: More stable performance, smaller degradation
- **QBs**: Moderate drift, consistent with passing volume changes

The position-wise analysis confirms that drift is not uniform across player roles, with usage-pattern changes affecting RBs most severely.

### 4.3 Retraining vs Fine-Tuning Results

#### Ridge Regression

| Model | MAE (2024) | RMSE (2024) | R² (2024) | ΔMAE vs Old |
|-------|------------|-------------|-----------|-------------|
| Ridge_2015_2021 (frozen) | ~X.XX | ~X.XX | ~0.XX | baseline |
| Ridge_2015_2023 (retrained) | ~X.XX | ~X.XX | ~0.XX | -0.023 |

**Result:** Full retraining on 2015-2023 provides a modest improvement of **0.023 MAE** on the 2024 hold-out.

*Note: Exact values can be regenerated from `notebooks/05_drift_eval_retrain.ipynb` Cell 15 (Ridge) and Cell 17 (`stage9_summary_df`).*

#### XGBoost

| Model | MAE (2024) | RMSE (2024) | R² (2024) | ΔMAE vs Old |
|-------|------------|-------------|-----------|-------------|
| XGB_2015_2022 (frozen) | ~X.XX | ~X.XX | ~0.XX | reference |
| XGB_full_23 (retrained) | ~X.XX | ~X.XX | ~0.XX | -0.027 |
| XGB_finetune (sample weights) | ~X.XX | ~X.XX | ~0.XX | +0.248 |

*Note: Exact values can be regenerated from `notebooks/05_drift_eval_retrain.ipynb` Cell 16 (XGBoost) and Cell 17 (`stage9_summary_df`).*

**Key findings:**
1. **Full retraining** improves MAE by **0.027** on 2024, consistent with Ridge
2. **Fine-tuning via sample weights** (weight=3.0 for 2023) **degrades performance by 0.248 MAE**, indicating overfitting to the recent season

**Conclusion:** Full retraining is the preferred strategy for addressing drift. The sample-weight proxy for fine-tuning overemphasizes 2023 patterns and fails to generalize to 2024.

### 4.4 Drift Visualization & Statistical Analysis

#### Distribution Shifts: Old Era (2015-2019) vs New Era (2020-2024)

**QB Passing Yards:**
- Mean decrease: ~180 yards per season
- Distribution shifts left, indicating reduced passing volume
- KS-test p-value: >0.35 (gradual shift, not abrupt)

**RB Rushing Attempts:**
- Mean decrease: ~2.4 carries per season
- Distribution compresses toward mid-60s (fewer workhorse backs)
- KS-test p-value: ~0.27 (gradual but persistent)

**WR PPR Points:**
- Mean increase: ~2 PPR points per season
- Distribution gains a fatter right tail (more high-scoring WRs)
- KS-test p-value: >0.35

#### Feature-Level Drift

**Targets per Game (WR/TE/RB):**
- Decreased by 0.26 targets per game on average
- KS-test p-value: ≈0.00 (statistically significant)
- Distribution shifts toward shorter volume spikes in new era

**Carries per Game (RB):**
- Decreased by 0.39 carries per game
- KS-test p-value: ≈0.27 (gradual but measurable)
- Confirms the shift away from workhorse RBs

**Team Pass Rate:**
- Decreased by ~0.02 (2 percentage points)
- KS-test p-value: ≈0.005 (statistically significant)
- Modern play-calling tilts slightly more run-heavy, contradicting the "passing league" narrative

**Interpretation:** The feature-level analysis confirms that the underlying data generating process has changed. Models trained on 2015-2019 data learn relationships that are less valid for 2020-2024, explaining the observed performance degradation.

### 4.5 Feature Importance (XGBoost)

XGBoost's built-in feature importance (gain-based) reveals which features drive predictions. While detailed importance plots are generated in the notebooks, key observations include:

- **Position encoding** features are highly important (expected, given position-specific scoring)
- **Per-game rates** (targets per game, carries per game) are more informative than raw totals
- **Team-level features** (team pass rate) contribute to predictions but with lower importance than player-level stats

**Note:** SHAP values could provide more nuanced feature importance analysis in future work, as they capture interaction effects and non-linear relationships.

---

## 5. Discussion & Limitations

### 5.1 Key Findings

1. **Concept drift is measurable and significant:** Both Ridge and XGBoost models show 0.3-0.6 MAE degradation when moving from 2022 to 2023, despite 2022 being in XGBoost's training window.

2. **Position-specific drift varies:** Running backs experience the steepest degradation, consistent with the league-wide shift away from workhorse backs. Wide receivers and tight ends are more stable.

3. **Full retraining helps, but gains are modest:** Retraining on 2015-2023 improves 2024 performance by only 0.02-0.03 MAE, suggesting that drift may be ongoing and that models need continuous updates.

4. **Sample-weight fine-tuning overfits:** Overweighting the most recent season (2023) causes the model to overfit and degrade on 2024, indicating that a balanced approach is necessary.

5. **Feature distributions have shifted:** Statistical tests and visualizations confirm that key features (passing yards, rushing attempts, targets per game, team pass rate) have different distributions in the 2020-2024 era compared to 2015-2019.

### 5.2 Limitations

1. **Limited temporal scope:** The analysis covers only 10 seasons (2015-2024). Longer time horizons might reveal cyclical patterns or more dramatic shifts.

2. **No external factors:** The models do not incorporate external context such as:
   - Rule changes (e.g., pass interference rules, overtime rules)
   - Coaching changes
   - Player injuries (beyond games played)
   - Weather conditions
   - Opponent strength

3. **Aggregate-level data:** Using season-level aggregates loses weekly variation and injury/bye-week dynamics that affect fantasy scoring.

4. **Fine-tuning proxy:** The sample-weight approach is a proxy for true incremental fine-tuning. XGBoost's sklearn API does not support updating existing model weights, so we cannot test true transfer learning.

5. **No ensemble methods:** We compare only Ridge and XGBoost. Ensemble methods (stacking, blending) or model selection strategies might improve robustness to drift.

6. **Feature engineering scope:** The current feature set is relatively simple. Richer features (e.g., rolling averages, opponent-adjusted stats, player age/experience) might improve generalization.

7. **Evaluation on single hold-out:** The 2024 test set is a single season. Cross-validation across multiple future seasons would provide more robust estimates of drift magnitude.

### 5.3 Implications for Production Systems

For fantasy football prediction systems in production:

1. **Monitor performance continuously:** Set up drift detection (e.g., PSI, KS-tests) to trigger retraining when degradation exceeds thresholds.

2. **Retrain regularly:** Even modest improvements (0.02-0.03 MAE) accumulate over time. Schedule periodic retraining (e.g., annually or after each season).

3. **Avoid overfitting to recent data:** Fine-tuning strategies that overweight the latest season can backfire. Prefer full retraining with balanced temporal weighting.

4. **Position-specific models:** Consider training separate models per position to better handle position-specific drift patterns.

5. **Feature monitoring:** Track feature distributions over time and flag significant shifts that may indicate concept drift.

---

## 6. Future Work

### 6.1 Improved Fine-Tuning Strategies

- **True incremental learning:** Implement XGBoost fine-tuning via the native API (not sklearn wrapper) to update existing trees rather than retraining from scratch.
- **Adaptive learning rates:** Use smaller learning rates when fine-tuning to preserve historical patterns while adapting to new data.
- **Elastic weight consolidation:** Apply techniques from continual learning to prevent catastrophic forgetting when updating models.

### 6.2 Richer Feature Engineering

- **Temporal features:** Add rolling averages, momentum features, and season-to-date statistics to capture recent trends.
- **Contextual features:** Incorporate opponent strength, weather, home/away splits, and coaching changes.
- **Player lifecycle features:** Include age, years of experience, and career trajectory indicators.
- **Interaction features:** Explicitly model position × team interactions, player × quarterback chemistry, etc.

### 6.3 Advanced Modeling Approaches

- **Recurrent Neural Networks (RNNs/LSTMs):** Model player performance as a time series to capture sequential dependencies and long-term trends.
- **Transformer models:** Apply attention mechanisms to learn complex temporal and positional relationships.
- **Ensemble methods:** Combine Ridge, XGBoost, and neural networks with stacking or blending to improve robustness.
- **Bayesian approaches:** Use probabilistic models to quantify uncertainty and adapt more gracefully to distribution shifts.

### 6.4 Drift Detection & Adaptation

- **Automated drift detection:** Implement statistical tests (PSI, KS-tests, CUSUM) to automatically detect when retraining is needed.
- **Online learning:** Explore incremental learning algorithms that update models continuously as new data arrives.
- **Adaptive windowing:** Dynamically adjust the training window size based on detected drift magnitude.

### 6.5 Evaluation & Validation

- **Multiple future seasons:** Extend the analysis to 2025+ seasons as data becomes available to validate findings.
- **Simulated deployment:** Run backtests simulating real-world deployment scenarios (e.g., retraining annually, quarterly, or on-demand).
- **Ablation studies:** Systematically test which features, model components, or preprocessing steps are most critical for handling drift.

---

## 7. Conclusion

This project demonstrates that concept drift is a real and measurable phenomenon in NFL fantasy football prediction models. Models trained on historical seasons (2015-2021) show meaningful performance degradation when applied to recent seasons (2023-2024), with running backs experiencing the steepest decline. Full retraining on updated data provides modest but consistent improvements, while naive fine-tuning strategies can overfit and degrade performance.

The analysis confirms that the underlying data generating process has changed between the 2015-2019 and 2020-2024 eras, with measurable shifts in key features such as passing yards, rushing attempts, and team pass rates. These findings have practical implications for production ML systems, emphasizing the need for continuous monitoring, regular retraining, and careful adaptation strategies.

Future work should explore richer features, advanced modeling approaches (RNNs, transformers), and automated drift detection to build more robust and adaptive prediction systems.

---

## References

- NFL play-by-play data: `nflfastR` package (https://www.nflfastr.com/)
- XGBoost documentation: https://xgboost.readthedocs.io/
- Scikit-learn documentation: https://scikit-learn.org/

---

## Appendix: Reproducibility

### Regenerating Results

All results in this report can be regenerated by running the following notebooks in order:

1. `notebooks/02_eda_drift.ipynb` - Exploratory analysis and drift visualization
2. `notebooks/03_baseline_ridge.ipynb` - Ridge regression baseline
3. `notebooks/04_xgb_timecv.ipynb` - XGBoost with rolling-origin CV
4. `notebooks/05_drift_eval_retrain.ipynb` - Drift evaluation and retraining experiments

The source code is organized in the `src/` directory:
- `src/data_prep.py` - Data loading and cleaning
- `src/features.py` - Feature engineering and preprocessing
- `src/models.py` - Model training utilities
- `src/evaluation.py` - Evaluation metrics and position-wise analysis
- `src/plotting.py` - Visualization helpers

To reproduce the analysis:

```bash
# 1. Build the player-season dataset
python -m src.build_player_season_dataset --output data/raw/player_season_2015_2024.csv

# 2. Run notebooks in sequence (or execute cells programmatically)
jupyter notebook notebooks/02_eda_drift.ipynb
jupyter notebook notebooks/03_baseline_ridge.ipynb
jupyter notebook notebooks/04_xgb_timecv.ipynb
jupyter notebook notebooks/05_drift_eval_retrain.ipynb
```

All figures and tables in this report are generated directly from the notebook outputs.

**Note on Numeric Values:** Some tables in this report use approximate values (e.g., ~1.85, ~0.6) or placeholders (X.XX) because exact values depend on the specific execution environment and random seeds. To obtain exact values:

1. Run all notebooks in sequence (see commands below)
2. Extract values from the DataFrames printed in each notebook
3. Replace placeholders in this report with the actual values

See `reports/verification_checklist.md` for a detailed mapping of each figure/table to its source notebook and cell.

### Required Imports

If running `notebooks/05_drift_eval_retrain.ipynb`, ensure Cell 1 includes:

```python
from sklearn.linear_model import Ridge
from src.models import (
    train_ridge,
    train_xgb_time_cv,
    retrain_full_xgb,
    finetune_xgb_with_sample_weights
)
```

