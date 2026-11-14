# 5-method (+DCT-QIM) rerun

Finishes the half-done 5-method dataset run and retrains the AWSRE models on it.

| File | What it does |
|---|---|
| `five_method_config.py` | Same design as the dissertation run (α 10–30, 17 attack conditions, 32×32 mark, 2 marks per host, seed 2026). Methods: DCT, DWT, DCT-SVD, DWT-SVD, BLOCK-SVD, plus DCT-QIM unless `AWSRE_INCLUDE_DCT_QIM=0`. 400 + 400 hosts by default (`AWSRE_IMAGES_PER_DATASET`). |
| `run_five_method_dataset.py` | Parallel, resumable version of `scientific_dataset_runner.py`. Rows are keyed exactly like the original runner, so rows from the earlier partial run are skipped. |
| `train_five_method_models.py` | Same models, features, host-grouped split and Pareto + ideal-point rule as `train_performance_models.py`, plus fixed-strategy and "old 3 methods only" baselines. Writes `model_summary.json`, which the site shows. |
| `AWSRE_5method_rerun_colab.ipynb` | Runs all of it in Google Colab against the data on Google Drive and uploads the models to Hugging Face (`avaqifli/awsre-models-5methods`). |

The site (`core/recommender.py`) downloads `avaqifli/awsre-models-5methods` when it exists and falls back to the dissertation's 3-method models (`avaqifli/awsre-models`) otherwise.

Forest size: the clean-quality and strategy models use 200 trees and `min_samples_leaf=5` (the dissertation used 350 and 2) so the models fit the hosted site's memory. `AWSRE_COMPACT=0` restores the original settings.
