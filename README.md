# Road Closure Prediction API

A machine learning system that predicts whether a traffic incident will require a road closure, deployed as a REST API on Hugging Face Spaces.

---

Check out Live API at https://supratimkukri-RoadClosure.hf.space/

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## Overview

Given 7 core fields about a traffic incident (time, location, cause, priority, zone, corridor, and vehicle type), the system derives 35+ engineered features and predicts the probability of a road closure, returning a binary decision, a confidence score, and a risk level (Low / Medium / High).

---

## Architecture

```
CSV Data
   │
   ▼
┌───────────────────────────────────────────┐
│           Feature Engineering             │
│  ┌──────────────┐  ┌────────────────────┐ │
│  │ Categorical  │  │   Time Features    │ │
│  │  Cleaning    │  │ (cyclical + flags)─│ │
│  └──────────────┘  └────────────────────┘ │
│  ┌──────────────┐  ┌───────────────────┐  │
│  │ Geo Cluster  │  │   Statistical     │  │
│  │  (KMeans)    │  │  Target Encoding  │  │
│  └──────────────┘  └───────────────────┘  │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ Interaction  │  │  Domain Features │   │
│  │  Features    │  │  (15 heuristics) │   │
│  └──────────────┘  └──────────────────┘   │
└───────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│         Preprocessing & Encoding        │
│  LabelEncoder × 6 + SimpleImputer       │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│         SMOTEENN Resampling             │
│  (handles severe class imbalance)       │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│       Soft-Voting Ensemble Model        │
│  ┌──────────┐ ┌─────────┐ ┌──────────┐  │
│  │ XGBoost  │ │LightGBM │ │  Random  │  │
│  │(Optuna   │ │(Optuna  │ │  Forest  │  │
│  │ tuned)   │ │ tuned)  │ │ (Optuna  │  │
│  │          │ │         │ │  tuned)  │  │
│  └──────────┘ └─────────┘ └──────────┘  │
│         Optuna-tuned weights            │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│      Optimal Threshold Selection        │
│  Precision-Recall curve (recall ≥ 0.65  │
│  and precision ≥ 0.35 constraint)       │
└─────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────┐
│         FastAPI REST Server             │
│  POST /predict      (single)            │
│  POST /predict/batch (up to 100)        │
│  GET  /health                           │
└─────────────────────────────────────────┘
```

---

## Feature Engineering

The system transforms 7 raw user inputs into 40 features across six categories.

### 1. Categorical Cleaning

Raw categorical fields are standardised before encoding:

| Field | Cleaning Rule |
|---|---|
| `veh_type` | Empty strings → `"unknown"` |
| `zone` | `"NULL"` values → `"unknown"` |
| `corridor` | Nulls → `"Non-corridor"` |
| `event_cause` | Nulls → `"others"` |
| `priority` | `"NULL"` → `"Low"` |
| `event_type` | Nulls → `"unplanned"` |

### 2. Time Features

Extracted from `start_datetime`:

- **Raw:** `hour`, `day_of_week`, `month`
- **Cyclical encoding** (to capture periodicity without ordinal bias):
  - `hour_sin`, `hour_cos` — 24-hour cycle
  - `dow_sin`, `dow_cos` — 7-day cycle
  - `month_sin`, `month_cos` — 12-month cycle
- **Binary flags:**
  - `is_peak_hour` — hours 7–9 and 17–21
  - `is_weekend` — Saturday/Sunday
  - `is_night` — hours 22–23 and 0–4

### 3. Geographic Features

- **KMeans clustering** (`n_clusters=12`) fitted on `(latitude, longitude)` pairs during training, producing a `geo_cluster` label per incident. Cluster centroids are serialised in the model bundle and reused at inference.

### 4. Statistical / Target Encoding

Computed from training data and stored in the bundle:

| Feature | Description |
|---|---|
| `cause_closure_rate` | Mean road closure rate per `event_cause` |
| `corridor_closure_rate` | Mean road closure rate per `corridor` |
| `corridor_volume` | Total incident count per `corridor` (traffic density proxy) |

### 5. Interaction Features

| Feature | Formula |
|---|---|
| `cause_x_peak` | `cause_closure_rate × is_peak_hour` |
| `priority_x_unplanned` | `priority_num × (event_type == "unplanned")` |
| `cause_x_corridor_vol` | `cause_closure_rate × log1p(corridor_volume)` |
| `priority_num` | `High → 1`, `Low → 0` |

### 6. Domain Features (15 heuristics)

Causally motivated binary or continuous signals derived from optional fields:

| Feature | Logic |
|---|---|
| `is_heavy_vehicle` | `veh_type` contains truck/lorry/bus/tanker/trailer/heavy |
| `has_cargo` | `cargo_material` is not null |
| `duration_hours` | `end_datetime − start_datetime` (clipped 0–48 h) |
| `is_old_truck` | `age_of_truck > 10` years |
| `at_junction` | `junction` field is not null |
| `has_police` | `assigned_to_police_id` is not null |
| `is_accident` | `citizen_accident_id` is not null |
| `has_direction` | `direction` field is not null |
| `is_both_directions` | `direction` contains "both", "all", or "contra" |
| `is_serious_breakdown` | `reason_breakdown` contains fire/accident/collision/engine/oil/flood |
| `is_high_risk_zone` | `zone` contains highway/tunnel/bridge/school/hospital |
| `time_to_resolve_hours` | `resolved_datetime − start_datetime` (clipped 0–72 h) |
| `resolved_at_diff_location` | Resolved address differs from incident address |
| `has_comment` | `comment` field is not null |
| `has_metadata` | `meta_data` field is not null |

### Encoding & Imputation

After feature construction:

- Six categorical columns (`event_cause`, `priority`, `veh_type`, `corridor`, `zone`, `event_type`) are encoded with `LabelEncoder`. Unseen categories at inference receive value `-1`.
- All remaining missing values are filled with a `SimpleImputer` using median strategy.
- Both the encoders and the imputer are serialised in the model bundle.

---

## Model Architecture

### Ensemble: Soft-Voting Classifier

Three base models are combined via weighted soft voting (averaging predicted probabilities):

```
Final Probability = (w_xgb × P_xgb + w_lgbm × P_lgbm + w_rf × P_rf)
                    ───────────────────────────────────────────────────
                              w_xgb + w_lgbm + w_rf
```

#### Base Models

**XGBoost**
- Gradient-boosted decision trees
- Class imbalance handled via `scale_pos_weight` (≈ neg/pos ratio)
- Evaluated with `aucpr` metric (area under precision-recall curve)

**LightGBM**
- Gradient boosting with leaf-wise tree growth
- Class imbalance handled via `class_weight="balanced"`

**Random Forest**
- Bagged decision tree ensemble
- Class imbalance handled via `class_weight="balanced_subsample"`
- Parallelised with `n_jobs=-1`

### Class Imbalance Handling: SMOTEENN

Before training the final ensemble, the training set is resampled with **SMOTEENN**, which combines:
- **SMOTE** — generates synthetic minority-class (road closure) samples via nearest-neighbour interpolation
- **ENN (Edited Nearest Neighbours)** — removes noisy majority-class samples

This is also applied per fold during cross-validation to prevent data leakage.

---

## Training Pipeline

Training runs in six sequential phases, all driven by Optuna Bayesian hyperparameter optimisation:

```
Phase 1 — Tune XGBoost        (25 trials)
Phase 2 — Tune LightGBM       (25 trials)
Phase 3 — Tune Random Forest   (20 trials)
Phase 4 — Tune ensemble weights (15 trials)
Phase 5 — Retrain on full training set (with SMOTEENN)
Phase 6 — Select optimal classification threshold
```

### Cross-Validation Strategy

All tuning uses 5-fold `StratifiedKFold` CV. Within each fold:
1. SMOTEENN is applied only to the training split (never to the validation split)
2. The model is trained on the resampled data
3. Probabilities are predicted on the held-out fold
4. An optimal per-fold threshold is found via the precision-recall curve
5. Road Closure F1 is computed and averaged across folds

### Threshold Selection

After full retraining, the optimal threshold is chosen from the precision-recall curve on the held-out test set (20% split), subject to:
- Recall ≥ 0.65 (catches most actual closures)
- Precision ≥ 0.35 (limits false alarms)

If no threshold satisfies both constraints, the F1-maximising threshold is used as a fallback. The threshold is stored in the model bundle and applied at every inference call.

### Saved Model Bundle

Everything needed for inference is serialised into a single `.pkl` file:

```python
bundle = {
    "model":       VotingClassifier,   # fitted ensemble
    "encoders":    dict[str, LabelEncoder],  # one per categorical column
    "imputer":     SimpleImputer,      # median imputer
    "kmeans":      KMeans,             # geo cluster model
    "stats":       dict,               # cause/corridor rates and volumes
    "features":    list[str],          # ordered feature list (40 features)
    "threshold":   float,              # optimal classification threshold
    "best_params": dict,               # hyperparameters for all three models
}
```

The bundle is hosted on Hugging Face Hub (`SupratimKukri/road-closure-model`) and downloaded at API startup via `hf_hub_download`.

---

## API Reference

Base URL: `http://localhost:7860` (or your Hugging Face Space URL)

### `POST /predict`

Predicts road closure for a single incident.

**Required fields:**

| Field | Type | Example | Description |
|---|---|---|---|
| `start_datetime` | string | `"2026-06-19 08:30:00+0530"` | Incident start time |
| `latitude` | float | `12.97` | Incident latitude |
| `longitude` | float | `77.59` | Incident longitude |
| `event_cause` | string | `"vehicle_breakdown"` | Cause of incident |
| `priority` | string | `"High"` | `"High"` or `"Low"` |
| `zone` | string | `"East Zone 1"` | Administrative zone |
| `corridor` | string | `"ORR"` | Road corridor name |

**Optional fields** (improve accuracy when provided):
`event_type`, `veh_type`, `direction`, `junction`, `reason_breakdown`, `cargo_material`, `age_of_truck`, `comment`

**Response:**

```json
{
  "prediction": {
    "road_closure_required": true,
    "confidence": 0.812,
    "risk_level": "High"
  },
  "input_summary": { ... },
  "model_info": {
    "threshold_used": 0.423
  }
}
```

Risk levels: `"Low"` (confidence < 0.3) · `"Medium"` (0.3–0.6) · `"High"` (> 0.6)

### `POST /predict/batch`

Accepts a JSON array of up to 100 prediction requests. Returns results in the same order as the input.

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`.

---

## Deployment

The API is containerised and deployed on Hugging Face Spaces.

### Docker Setup

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1   # required by LightGBM

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "road_closure_app:app", "--host", "0.0.0.0", "--port", "7860"]
```

`libgomp1` is the OpenMP runtime required by LightGBM for multi-threaded tree building. The server starts on port 7860, which is the default port for Hugging Face Spaces.

### Model Loading

At startup, `road_closure_app.py` downloads `road_closure_model.pkl` from Hugging Face Hub:

```python
MODEL_PATH = hf_hub_download(
    repo_id="SupratimKukri/road-closure-model",
    filename="road_closure_model.pkl"
)
```

If the file is not found, the API starts in degraded mode and returns HTTP 503 on prediction endpoints.

---

## Project Structure

```
.
├── Final_Model.ipynb       # Training notebook (feature engineering + Optuna tuning)
├── road_closure_app.py     # FastAPI inference server
├── Dockerfile              # Container definition for HF Spaces deployment
├── requirements.txt        # Python dependencies
└── road_closure_model.pkl  # Serialised model bundle (hosted on HF Hub)
```

### Key Dependencies

| Package | Purpose |
|---|---|
| `xgboost` | XGBoost classifier |
| `lightgbm` | LightGBM classifier |
| `scikit-learn` | Random Forest, LabelEncoder, KMeans, imputer, CV |
| `imbalanced-learn` | SMOTEENN resampling |
| `optuna` | Bayesian hyperparameter optimisation |
| `fastapi` / `uvicorn` | REST API server |
| `huggingface_hub` | Model bundle download at startup |
| `pandas` / `numpy` | Feature engineering |
