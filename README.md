# Adaptive Predictive Maintenance with OEM Priors

Most bearing maintenance boils down to one of two strategies: replace on a fixed schedule (safe but wasteful) or wait for the vibration alarm to fire (reactive and too late for planning). This project builds a third option — start from the manufacturer's expected life curve, then adapt it in real time using a PID feedback controller on actual vibration data. The result is a bearing-specific remaining useful life estimate that updates continuously.

The interesting finding is in the regime analysis. The overall numbers favor a simple static curve over the adaptive models, but when you break results down by degradation phase, the regime-switching PID variant cuts prediction error by 6x during the mild damage phase — exactly the transition from "this bearing is fine" to "this bearing has started to fail." That's where maintenance decisions are hardest and where the adaptive approach earns its keep.

## What's in here

**Phase 1 — OEM spec extraction.** A RAG pipeline ingests ~500 pages of real SKF PDF documentation (the Rolling Bearings General Catalog, Bearing Damage and Failure Analysis guide, and Bearing Failures guide), chunks them with structure-aware parsing that handles dense product tables, embeds with MiniLM-L6-v2, and retrieves the specific load ratings and life parameters for the SKF 6205-2RS. The pipeline correctly extracts all key specs from a 354-page catalog — dynamic load rating, static load rating, bore diameter, life exponent — using a hybrid of semantic retrieval and exact text matching.

**Phase 2 — Adaptive drift modeling.** Vibration features (kurtosis, RMS, defect frequency energy) are extracted from the CWRU Bearing Data Center recordings and arranged into a synthetic degradation trajectory. A PID controller tracks the deviation between observed kurtosis and the OEM-derived baseline curve, producing an adapted remaining life estimate. A regime-switching variant monitors error volatility and increases PID gains when it detects the transition from normal wear to accelerated damage.

Five models are compared: threshold alarm, static exponential curve, rolling refit, PID adaptive, and PID + regime. Evaluation uses RMSE, MAE, detection lead time, false alarm rate, and the NASA asymmetric scoring function (which penalizes late predictions — missed failures — exponentially harder than early ones).

## Key results

| Phase | PID Adaptive MAE (hrs) | PID + Regime MAE (hrs) |
|-------|----------------------|----------------------|
| Healthy | 1,412 | 1,412 |
| Mild (0.007") | 779 | **125** |
| Moderate (0.014") | 48 | 50 |
| Severe (0.021") | 47 | 49 |

The regime predictor's value is concentrated at the mild damage transition — after that, both models converge because the signal is obvious. All models achieve 100% detection success with zero false alarms.

## Data

**CWRU Bearing Data Center**: Vibration recordings from an SKF 6205-2RS JEM deep groove ball bearing on a 2 HP motor test rig at ~1,750 RPM. Pre-seeded faults at 0.007", 0.014", and 0.021" on inner race, outer race, and rolling elements. 12,000 Hz sampling rate. Downloaded by `src/download_cwru.py`.

**SKF OEM documentation**: Real PDFs from SKF's public catalog and failure analysis guides. Not included in the repo — download with `scripts/download_oem_docs.sh`.

## Project structure

```
data/
  raw/              # CWRU .mat files (downloaded by script)
  processed/        # Extracted feature time series
  oem/              # SKF PDFs (downloaded by script)

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
  02_feature_exploration.ipynb # CWRU data and feature analysis
  03_model_comparison.ipynb    # Full model comparison

reports/
  adaptive_maintenance.Rmd    # Final report
  figures/

analysis/                     # All numeric outputs as CSV/JSON
```

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_oem_docs.sh
python src/download_cwru.py
```

Then run the notebooks in order, or knit the report directly.

## Limitations

The synthetic degradation trajectory (concatenating discrete fault severities) creates step-change transitions that penalize adaptive models and don't reflect real bearing degradation. The CWRU dataset was chosen for its direct connection to documented SKF bearing specs, but validation on true run-to-failure data (IMS, FEMTO/PRONOSTIA) is the obvious next step.

## Related

This uses the same PID adaptive drift framework from [adaptive-drift-forecasting](https://github.com/cjbrant/adaptive-drift-forecasting), applied to a domain where the physics are cleaner and the stakes are more concrete.
