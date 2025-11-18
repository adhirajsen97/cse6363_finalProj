# ML NFL Concept Drift Project

## Project Overview
NFL and fantasy-football analytics rely on models trained on historical player stats. But the league changes over time: coaching trends, rule changes, and player usage patterns shift, so models trained on old seasons gradually degrade. This phenomenon is **concept drift**.

This project will:
- Quantify how much a historical model’s performance decays on recent seasons.
- Retrain / fine‑tune the model on contemporary data.
- Analyze which positions and features are most affected by drift.

### Project Category
**Retraining of an existing model on new data** (per course project categories). If a pre‑trained model with weights is unavailable, the repository recreates a comparable baseline model on older seasons and then treats that as the “existing” model to retrain.

### Main Research Questions
1. How much does a player‑stat prediction model’s performance degrade when applied to newer NFL seasons?
2. Can fine‑tuning on limited new data restore or improve performance?
3. Which player positions (QB/RB/WR/TE) and which features are most affected by drift?
4. How do tree‑based ensembles compare to a simpler baseline (linear regression or Random Forest)?

## Project Structure
```
ml-nfl-drift/
  data/
    raw/
    processed/
  notebooks/
  src/
    __init__.py
    __main__.py
    data_prep.py
    features.py
    models.py
    evaluation.py
    plotting.py
  reports/
  requirements.txt
  README.md
```

- `data/raw/` and `data/processed/` hold input and cleaned datasets (git-ignored).
- `notebooks/` contains exploratory analysis.
- `src/` stores reusable Python modules and the module entry point (`python -m src`).
- `reports/` is a placeholder for generated figures and summaries.

## Installation
1. Create and activate a fresh virtual environment (conda or `python -m venv`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify the package imports cleanly:
   ```bash
   python -m src
   ```

You can now develop data pipelines, feature engineering, model training, and drift evaluation using the provided skeleton.
