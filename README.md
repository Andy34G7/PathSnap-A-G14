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
- `reports/eda/`: EDA tables, figures, Folium HTML maps, and [EDA_Summary_Report.md](reports/eda/EDA_Summary_Report.md)
- `src/features/` (Planned): Spatial projection, candidate generation, and feature extraction
- `src/models/` (Planned): Training and evaluation across 5 or more model families with Bayesian optimisation
- `src/hmm/` (Planned): HMM emission and transition modelling and Viterbi route reconstruction
- `src/api/` (Planned): FastAPI backend (Laptop 1)
- `src/ui/` (Planned): Evaluator interface (Laptop 2)
- `artifacts/` (Planned): Saved model weights, preprocessors, HMM configs, and experiment logs

## Setup and Usage
Run all commands from the repository root so `src.*` module imports resolve properly:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Run the EDA pipeline (requires train/ and student_test/ folders in repo root)
PYTHONPATH=. .venv/bin/python -m src.eda.inventory_and_quality
PYTHONPATH=. .venv/bin/python -m src.eda.trajectory_analysis
PYTHONPATH=. .venv/bin/python -m src.eda.network_and_route_analysis
PYTHONPATH=. .venv/bin/python -m src.eda.folium_visualisations
```
