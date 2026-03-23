# Adaptive Predictive Maintenance with OEM Priors

PID feedback control for bearing remaining useful life estimation, benchmarked across five public datasets, three bearing manufacturers, and three tiers of OEM prior quality. A RAG pipeline extracts the bearing specs automatically from manufacturer PDF catalogs.

## Problem

Bearings get replaced on a schedule derived from the manufacturer's L10 life rating. L10 is a fleet statistic. It doesn't account for installation quality, contamination, or load changes on a specific machine. Most bearings have life left at replacement; the ones that don't fail between scheduled swaps at 5-10x the cost.

This project builds a third option between scheduled replacement and waiting for the vibration alarm: start from the OEM's expected life curve, then adapt it in real time using a PID controller on actual vibration data. The result is a bearing-specific remaining useful life estimate that updates continuously, requires zero training data, and runs in constant time per observation.

## How it works

**OEM spec extraction.** A RAG pipeline ingests manufacturer PDF catalogs (SKF, Rexnord, LDK), chunks them with structure-aware parsing, embeds with MiniLM-L6-v2, and retrieves the dynamic load rating and life exponent for a given bearing designation. Handles dense product tables, scanned catalogs via OCR, and multi-variant disambiguation. Each extraction gets a confidence tier (high/medium/low/fallback) for human review.

**Adaptive drift model.** Vibration features (kurtosis, RMS, defect frequency energy) are extracted from sensor recordings. A PID controller tracks deviation between observed degradation and the OEM baseline curve, producing an adapted RUL estimate at each measurement. A regime-switching variant monitors error volatility and increases PID gains when it detects the transition from normal wear to accelerated damage.

**Evaluation.** Five models compared (threshold alarm, static exponential curve, rolling refit, PID adaptive, PID+regime) using RMSE, MAE, detection lead time, false alarm rate, and the NASA asymmetric scoring function.

## Datasets

| Dataset | Equipment | Manufacturer | Prior Quality | Trajectories |
|---------|-----------|-------------|---------------|-------------|
| IMS | Roller bearing | Rexnord ZA-2115 | Exact OEM | 4 failures |
| XJTU-SY | Ball bearing | LDK UER204 | Exact OEM | 15 run-to-failure |
| FEMTO | Ball bearing | Unknown (≈SKF 6204) | Approximate | 17 trajectories |
| C-MAPSS | Turbofan engine | N/A (fleet prior) | Fleet-derived | 200 engines |
| CWRU | Ball bearing | SKF 6205-2RS | Exact OEM | Synthetic reference |

## Key results

On real run-to-failure bearing data, PID variants beat static baselines by 18-20% RMSE on IMS. The regime-switching variant reduced RMSE by an additional 26.7% on XJTU-SY, concentrated at the mild-to-severe damage transition where maintenance decisions are hardest.

The RAG pipeline extracted specs from three manufacturers' catalogs with no fallbacks. Two of four extractions hit within 7% of ground truth. The framework generalizes across manufacturers without any manufacturer-specific tuning.

On smooth simulated degradation (C-MAPSS), the static curve wins because the degradation is well-described by a simple function. The PID framework earns its keep on real nonlinear degradation and stays out of the way on smooth trends.

## Project structure

```
data/
  raw/              # Vibration recordings (downloaded by scripts)
  processed/        # Extracted feature time series
  oem/              # Manufacturer PDFs (downloaded by script)

src/
  download_cwru.py        # Fetch CWRU vibration data
  feature_extraction.py   # Vibration signal → degradation features
  oem_prior.py            # L10 life calculation and OEM baseline curve
  adaptive_drift.py       # PID adaptive drift model
  baselines.py            # Static curve, rolling refit, threshold alarm
  regime_predictor.py     # Error-volatility regime detection
  evaluation.py           # RUL metrics, detection metrics, model comparison

rag/
  pdf_extract.py    # PyMuPDF text extraction with table reconstruction
  ingest.py         # Structure-aware chunking and ChromaDB embedding
  retrieve.py       # Hybrid semantic + text-matching retrieval
  extract_params.py # Parse retrieved chunks into structured parameters

notebooks/
  01_oem_extraction.ipynb     # RAG pipeline walkthrough
  02_feature_exploration.ipynb # Feature analysis per dataset
  03_model_comparison.ipynb    # Full benchmark comparison

reports/
  benchmark_report.Rmd    # Multi-dataset benchmark report
  figures/

analysis/                 # All numeric outputs as CSV/JSON
```

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_oem_docs.sh
python src/download_cwru.py
python src/download_ims.py
python src/download_xjtu.py
```

Then run the notebooks in order, or knit the report directly.

## Limitations

All bearing datasets use accelerated degradation with loads chosen to produce failure in hours or days. The model tracks a single feature (kurtosis) on a single axis. Sample sizes are small (4-15 bearings per dataset). These comparisons are directional, not statistically definitive. See the benchmark report for full discussion.

## Related

This extends the PID adaptive drift framework from [adaptive-drift-forecasting](https://github.com/cjbrant/adaptive-drift-forecasting) into a domain where the physics are cleaner and the stakes are more concrete.
