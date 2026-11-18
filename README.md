# PathSnap: Group A-G14

**UE24CS352A: Machine Learning Hackathon**  
Distributed map-matching and route reconstruction using candidate classification with hybrid HMM and Viterbi decoding.

## Team (A-G14)
- `Andy34G7`
- `Abhiroop-24`
- `A6h1ramahy`
- `Neo-pixel123`

## Summary Results (student_test, 10 Trips, 19,160 GPS Points)
- **Candidate Recall (`K = 8`):** `99.6138%` (`0.3862%` miss rate)
- **Tuned LightGBM Classifier Only:** `94.5929%` Top-1 Accuracy, `0.9674` Macro F1, `0.0404` Log Loss (`0.9505` 5-fold GroupKFold CV Macro F1)
- **Hybrid HMM/Viterbi + Route Boundary Realignment:** **`97.7036%` Observation Top-1 Accuracy**, **`96.3390%` Mean Route Arc F1** (`99.0493%` recall, `93.9500%` precision), **`99.7940%` Reconstructed Sequence Continuity** (`7 of 10` trips at `100.00%`, `0` Dijkstra truncations)

## Project Structure
- `src/eda/`: Dataset inspection, summary statistics, plots, and Folium visualisations
- `src/features/`: Dynamic UTM projection, turnaround-aware 20 m spatial-chord smoothing, 3-point `cKDTree` candidate generation, monotonic DP alignment, and 24-feature extraction
- `src/models/`: 6-model family benchmark (5-fold `GroupKFold` CV) and Optuna TPE Bayesian hyperparameter optimisation
- `src/hmm/`: Hybrid Newson-Krumm HMM + Viterbi decoder, shortest-path route reconstruction, Stage 4 continuous route boundary realignment, incremental accuracy ablation, and inference-only `predict_trip_route` helper
- `artifacts/`: Tracked LightGBM weights (`best_lgbm_tuned.joblib`, `untuned_lgbm.joblib` matching `subsample_freq=1`) and `hmm_config.json`
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
