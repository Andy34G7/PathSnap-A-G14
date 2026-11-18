# PathSnap: Group A-G14

**UE24CS352A: Machine Learning Hackathon**  
Distributed map-matching and route reconstruction using candidate classification with hybrid HMM and Viterbi decoding.

## Team (A-G14)
- `Andy34G7`
- `Abhiroop-24`
- `A6h1ramahy`
- `Neo-pixel123`

## Summary Results

### `student_test` (10 Trips, 19,160 GPS Points)
- **Candidate Recall (`K = 8`):** `99.6138%` (`0.3862%` miss rate)
- **Tuned LightGBM Classifier Only:** `94.5929%` Top-1 Accuracy, `0.9674` Macro F1, `0.0404` Log Loss (`0.9505` 5-fold GroupKFold CV Macro F1)
- **Hybrid HMM/Viterbi + Route Boundary Realignment:** **`97.7036%` Observation Top-1 Accuracy**, **`96.3390%` Mean Route Arc F1** (`99.0493%` recall, `93.9500%` precision), **`99.7940%` Reconstructed Sequence Continuity** (`7 of 10` trips at `100.00%`, `0` Dijkstra truncations)

### Blind Holdout `evaluator_test` (10 Trips, 19,674 GPS Points, Frozen Model)
- **Candidate Recall (`K = 8`):** `99.5578%`
- **Tuned LightGBM Classifier Only:** `95.2628%` Observation Top-1 Accuracy (`83.5606%` Mean Route Arc F1)
- **Hybrid HMM/Viterbi + Route Boundary Realignment:** **`97.5145%` Observation Top-1 Accuracy**, **`92.1917%` Mean Route Arc F1** (`99.3346%` recall, `87.8684%` precision, `87.3165%` Jaccard), **`100.0000%` Reconstructed Sequence Continuity across all 10 trips**, **`0` Dijkstra truncations**, **`7.8133 s` total inference time**

## Project Structure
- `src/eda/`: Dataset inspection, summary statistics, plots, and Folium visualisations
- `src/features/`: Dynamic UTM projection, turnaround-aware 20 m spatial-chord smoothing, 3-point `cKDTree` candidate generation, monotonic DP alignment, and 24-feature extraction
- `src/models/`: 6-model family benchmark (5-fold `GroupKFold` CV) and Optuna TPE Bayesian hyperparameter optimisation
- `src/hmm/`: Hybrid Newson-Krumm HMM + Viterbi decoder, shortest-path route reconstruction, Stage 4 continuous route boundary realignment, incremental accuracy ablation, and inference-only `predict_trip_route` helper
- `src/api/`: Laptop 1 FastAPI server (`server.py`) hosting frozen model weights, strict zero-leakage inference, post-prediction server-side evaluation, CMD response logging, and batch holdout runner
- `src/ui/`: Laptop 2 minimal web evaluator interface (`app.py`) for trip selection, `.track`/`.nodes`/`.arcs` payload transmission, evaluation metrics tables, and interactive Folium map viewing
- `artifacts/`: Tracked LightGBM weights (`best_lgbm_tuned.joblib`, `untuned_lgbm.joblib` matching `subsample_freq=1`) and `hmm_config.json`
- `reports/`: Consolidated technical report ([Phase1_Technical_Report.md](reports/Phase1_Technical_Report.md), [Phase1_Technical_Report.pdf](reports/Phase1_Technical_Report.pdf), [Final_Technical_Report.pdf](reports/Final_Technical_Report.pdf)), Phase 1 CSVs/figures, and Phase 2 `reports/evaluation/` outputs

## Setup and Pipeline Execution
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

# 4. Blind Holdout CLI Evaluation (runs frozen model over student_test and evaluator_test)
PYTHONPATH=. .venv/bin/python -m src.api.server --evaluate-only
```

## Two-Laptop LAN Deployment

### Laptop 1 (Model Server & Server-Side Evaluation)
1. Find the LAN IP address on Laptop 1 (`hostname -I` or `ipconfig`).
2. Start the FastAPI backend listening on all interfaces (`0.0.0.0:8000`):
```bash
PYTHONPATH=. .venv/bin/python -u -m src.api.server --host 0.0.0.0 --port 8000
```
3. Watch the Laptop 1 terminal for structured `[PathSnap API]` inference and post-prediction evaluation logs (`Top1`, `RouteF1`, `SeqCont`, `Error Buckets`, `Predicted Route`).

### Laptop 2 (Evaluator Web Interface)
1. Point `--api-url` to Laptop 1's LAN IP (use `http://127.0.0.1:8000` for single-machine testing):
```bash
PYTHONPATH=. .venv/bin/python -u -m src.ui.app --host 0.0.0.0 --port 8501 --api-url http://<LAPTOP1_LAN_IP>:8000
```
2. Open `http://127.0.0.1:8501` in a browser on Laptop 2, choose `evaluator_test` or `student_test`, select any trip (keeping `Send .track/.nodes/.arcs payload` checked to transmit raw geometry/GPS without `.route` over HTTP), and click **Reconstruct Trip** or **Evaluate All 10 Trips**.
