# PathSnap: Group A-G14

**UE24CS352A: Machine Learning Hackathon**  
Distributed map-matching and route reconstruction using candidate classification with HMM and Viterbi decoding.

## Team (A-G14)
- `Andy34G7`
- `Abhiroop-24`
- `A6h1ramahy`
- `Neo-pixel123`

## Project Structure
- `src/eda/`: Dataset inspection, summary statistics, plots, and Folium visualisations
- `src/features/`: Dynamic UTM projection, 3-point `cKDTree` candidate generation, monotonic DP alignment, and 24-feature extraction
- `src/models/`: 6-model family benchmark (5-fold `GroupKFold` CV) and Optuna TPE Bayesian hyperparameter optimisation
- `src/hmm/`: Directed Newson-Krumm HMM + Viterbi decoder, shortest-path route reconstruction, and inference API helper
- `artifacts/`: Saved untuned/tuned LightGBM weights (`best_lgbm_tuned.joblib`, `untuned_lgbm.joblib`) and `hmm_config.json`
- `reports/`: Consolidated [Phase1_Technical_Report.md](reports/Phase1_Technical_Report.md) / [Phase1_Technical_Report.pdf](reports/Phase1_Technical_Report.pdf), CSV logs, figures, and Folium HTML maps
- `src/api/` (Planned, Phase 2): FastAPI backend (Laptop 1)
- `src/ui/` (Planned, Phase 2): Evaluator interface (Laptop 2)

## Setup and Usage
Run all commands from the repository root so `src.*` module imports resolve properly:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 1. Exploratory Data Analysis (requires train/ and student_test/ folders in repo root)
PYTHONPATH=. .venv/bin/python -m src.eda.inventory_and_quality
PYTHONPATH=. .venv/bin/python -m src.eda.trajectory_analysis
PYTHONPATH=. .venv/bin/python -m src.eda.network_and_route_analysis
PYTHONPATH=. .venv/bin/python -m src.eda.folium_visualisations

# 2. Candidate Generation & 24-Feature Engineering
PYTHONPATH=. .venv/bin/python -m src.features.candidate_generation
PYTHONPATH=. .venv/bin/python -m src.features.feature_engineering

# 3. 6-Model Benchmark, Bayesian Optimisation & HMM/Viterbi Decoding
PYTHONPATH=. .venv/bin/python -m src.models.train_and_compare
PYTHONPATH=. .venv/bin/python -m src.models.bayesian_optimisation
PYTHONPATH=. .venv/bin/python -m src.hmm.viterbi_decoder
```
