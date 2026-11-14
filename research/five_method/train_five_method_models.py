"""Retrain the AWSRE performance models on the 5(+1)-method dataset.

Same models, features, targets, host-grouped 80/20 split and Pareto +
ideal-point selection rule as experiments/train_performance_models.py (the
dissertation's 3-method training). Differences, all reported in the
output so nothing is hidden:

* Forest size. With twice as many strategies the original settings
  (350 trees, min_samples_leaf=2) would give ~1 GB per model, too large
  for the hosted website. By default the clean-quality and strategy
  models use 200 trees and min_samples_leaf=5 (AWSRE_COMPACT=0 restores the
  original settings). The attack model already used the compact settings.
* Extra evaluation, to compare honestly with the old numbers:
    - Pareto hit rate of every FIXED strategy (always the same method and
      alpha) on the same test contexts, and the best of them;
    - AWSRE restricted to the 3 old methods, re-scored on the new data;
    - how often AWSRE picks each method.
  Written to model_summary.json next to the models (the website reads it).

Run from the repository root:
    python research/five_method/train_five_method_models.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for extra in (PROJECT_ROOT, HERE):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from five_method_config import CSV_PATH, OUTPUT_DIR, RANDOM_SEED  # noqa: E402

MODEL_DIR = Path(os.environ.get("AWSRE_MODEL_DIR", str(OUTPUT_DIR / "models")))
COMPACT = os.environ.get("AWSRE_COMPACT", "1") == "1"
MAIN_TREES, MAIN_LEAF = (200, 5) if COMPACT else (350, 2)
OLD_METHODS = ["DCT", "DCT-SVD", "DWT"]

HOST_FEATURES = [
    "host_width", "host_height", "host_resolution", "host_aspect_ratio",
    "host_brightness", "host_contrast", "host_entropy", "host_edge_density",
    "host_texture_variance", "host_frequency_energy", "host_high_frequency_ratio",
    "host_dynamic_range", "host_noise_level", "host_gradient_complexity",
    "host_histogram_uniformity", "host_smooth_area_ratio", "host_image_complexity_score",
]
WATERMARK_FEATURES = [
    "watermark_density", "watermark_entropy", "watermark_edge_complexity",
    "watermark_connected_components", "watermark_compactness", "watermark_fill_ratio",
    "watermark_symmetry_score", "watermark_stroke_complexity",
    "watermark_structural_complexity", "watermark_complexity_score",
]
STRATEGY_FEATURES = ["alpha", "method"]
ATTACK_FEATURES = ["attack_parameter_numeric", "attack_type"]
CONTEXT = ["dataset", "image_hash", "watermark_hash"]
OBJECTIVES = ["clean_mse", "mean_attack_ber", "embedding_time_seconds"]


def make_pipeline(numeric: List[str], categorical: List[str], n_estimators: int, min_samples_leaf: int) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
    )
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, random_state=RANDOM_SEED, n_jobs=-1,
        min_samples_leaf=min_samples_leaf, max_features=0.8,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def split_by_host(df: pd.DataFrame, test_size: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
    train_index, test_index = next(splitter.split(df, groups=df["image_hash"]))
    return df.iloc[train_index].copy(), df.iloc[test_index].copy()


def evaluate_model(name, model, train, test, features, targets):
    model.fit(train[features], train[targets])
    predictions = np.asarray(model.predict(test[features])).reshape(len(test), -1)
    rows = []
    for index, target in enumerate(targets):
        truth = test[target].to_numpy(dtype=float)
        estimate = predictions[:, index]
        rows.append({
            "experiment": name, "target": target,
            "train_rows": len(train), "test_rows": len(test),
            "train_hosts": train["image_hash"].nunique(), "test_hosts": test["image_hash"].nunique(),
            "MAE": mean_absolute_error(truth, estimate),
            "RMSE": mean_squared_error(truth, estimate) ** 0.5,
            "R2": r2_score(truth, estimate),
        })
    return model, rows


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "image_hash", "watermark_hash", "method", "alpha"]
    return (df.sort_values(keys).drop_duplicates(keys)
            .dropna(subset=["clean_psnr", "clean_ssim", "clean_mse"]).copy())


def build_strategy_robustness(df: pd.DataFrame) -> pd.DataFrame:
    strategy = ["dataset", "image_hash", "image_name", "watermark_hash", "watermark_name", "method", "alpha"]
    attacked = df[df["attack_type"] != "none"]
    robustness = attacked.groupby(strategy, as_index=False).agg(
        mean_attack_ber=("attack_ber", "mean"),
        worst_attack_ber=("attack_ber", "max"),
        mean_attack_correlation=("attack_correlation", "mean"),
        n_attacks=("attack_ber", "count"),
    )
    clean = build_clean_dataset(df)[
        strategy + HOST_FEATURES + WATERMARK_FEATURES
        + ["clean_mse", "clean_psnr", "clean_ssim", "embedding_time_seconds"]
    ]
    return clean.merge(robustness, on=strategy, how="inner")


def pareto_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            keep[i] = False
    return keep


def utopia_pick(objectives: np.ndarray) -> int:
    mask = pareto_mask(objectives)
    indexes = np.where(mask)[0]
    candidate = objectives[indexes]
    low, high = candidate.min(axis=0), candidate.max(axis=0)
    span = high - low
    span[span == 0] = 1.0
    return int(indexes[int(np.argmin(np.linalg.norm((candidate - low) / span, axis=1)))])


def regret(actual: np.ndarray, actual_mask: np.ndarray, selected: int) -> float:
    low, high = actual.min(axis=0), actual.max(axis=0)
    span = high - low
    span[span == 0] = 1.0
    pareto = (actual[actual_mask] - low) / span
    chosen = (actual[selected] - low) / span
    return float(np.min(np.linalg.norm(pareto - chosen, axis=1)))


def evaluate_recommendations(model, test: pd.DataFrame, features: List[str], strategies: List[Tuple[str, float]]):
    """Pareto hit rate of AWSRE (all methods and old-3-methods only), of every
    fixed strategy, and of the oracle, on test contexts that have every
    strategy measured."""
    evaluation = test.copy()
    predicted = np.asarray(model.predict(evaluation[features]))
    evaluation["pred_clean_mse"] = predicted[:, 0]
    evaluation["pred_mean_attack_ber"] = predicted[:, 1]
    evaluation["pred_embedding_time"] = predicted[:, 2]
    pred_cols = ["pred_clean_mse", "pred_mean_attack_ber", "pred_embedding_time"]

    strategy_index = {s: i for i, s in enumerate(strategies)}
    fixed_hits = np.zeros(len(strategies))
    details = []
    skipped = 0
    for context, group in evaluation.groupby(CONTEXT, sort=False):
        group = group.copy()
        group["strategy"] = list(zip(group["method"], group["alpha"].astype(float)))
        group = group.drop_duplicates("strategy")
        if len(group) != len(strategies):
            skipped += 1
            continue
        group = group.set_index("strategy").loc[strategies].reset_index()
        actual = group[OBJECTIVES].to_numpy(dtype=float)
        mask = pareto_mask(actual)
        fixed_hits += mask

        selected = utopia_pick(group[pred_cols].to_numpy(dtype=float))
        old = np.where(group["method"].isin(OLD_METHODS).to_numpy())[0]
        old_actual = actual[old]
        old_mask = pareto_mask(old_actual)
        old_selected_local = utopia_pick(group.iloc[old][pred_cols].to_numpy(dtype=float))
        oracle = utopia_pick(actual)

        details.append({
            "dataset": context[0], "image_hash": context[1], "watermark_hash": context[2],
            "selected_method": group.iloc[selected]["method"],
            "selected_alpha": float(group.iloc[selected]["alpha"]),
            "actual_pareto_hit": bool(mask[selected]),
            "regret": regret(actual, mask, selected),
            "oracle_method": group.iloc[oracle]["method"],
            "oracle_alpha": float(group.iloc[oracle]["alpha"]),
            "old3_selected_method": group.iloc[old[old_selected_local]]["method"],
            "old3_selected_alpha": float(group.iloc[old[old_selected_local]]["alpha"]),
            "old3_pareto_hit_within_old3": bool(old_mask[old_selected_local]),
            "old3_pareto_hit_within_all": bool(mask[old[old_selected_local]]),
            "actual_clean_psnr": float(group.iloc[selected]["clean_psnr"]),
            "actual_mean_attack_ber": float(group.iloc[selected]["mean_attack_ber"]),
            "actual_embedding_time": float(group.iloc[selected]["embedding_time_seconds"]),
        })

    details_df = pd.DataFrame(details)
    n = len(details_df)
    fixed = pd.DataFrame({
        "method": [s[0] for s in strategies], "alpha": [s[1] for s in strategies],
        "pareto_hit_rate": fixed_hits / max(n, 1),
    }).sort_values("pareto_hit_rate", ascending=False).reset_index(drop=True)
    summary = {
        "contexts": n,
        "contexts_skipped_incomplete": skipped,
        "pareto_hit_rate": float(details_df["actual_pareto_hit"].mean()) if n else None,
        "mean_regret": float(details_df["regret"].mean()) if n else None,
        "zero_regret_rate": float(details_df["regret"].eq(0).mean()) if n else None,
        "best_fixed_strategy": {
            "method": fixed.iloc[0]["method"], "alpha": float(fixed.iloc[0]["alpha"]),
            "pareto_hit_rate": float(fixed.iloc[0]["pareto_hit_rate"]),
        } if n else None,
        "old3_only_pareto_hit_rate_within_old3": float(details_df["old3_pareto_hit_within_old3"].mean()) if n else None,
        "old3_only_pareto_hit_rate_within_all": float(details_df["old3_pareto_hit_within_all"].mean()) if n else None,
        "awsre_pick_share_by_method": details_df["selected_method"].value_counts(normalize=True).round(4).to_dict() if n else {},
        "oracle_pick_share_by_method": details_df["oracle_method"].value_counts(normalize=True).round(4).to_dict() if n else {},
        "distinct_awsre_picks": int(details_df[["selected_method", "selected_alpha"]].drop_duplicates().shape[0]) if n else 0,
    }
    return summary, details_df, fixed


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {CSV_PATH}", flush=True)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["status"] == "SUCCESS"].copy()
    df["attack_parameter_numeric"] = pd.to_numeric(df["attack_parameter"], errors="coerce").fillna(0.0)
    df["alpha"] = df["alpha"].astype(float)
    missing = [c for c in HOST_FEATURES + WATERMARK_FEATURES + STRATEGY_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError("Missing columns: " + ", ".join(missing))

    methods = sorted(df["method"].dropna().unique().tolist())
    alphas = sorted(df["alpha"].dropna().unique().tolist())
    strategies = [(m, a) for m in methods for a in alphas]
    print(f"Rows: {len(df)} | hosts: {df['image_hash'].nunique()} | methods: {methods} | alphas: {alphas}")
    print(f"Forest settings for clean/strategy models: {MAIN_TREES} trees, min_samples_leaf={MAIN_LEAF}")

    context_features = HOST_FEATURES + WATERMARK_FEATURES + STRATEGY_FEATURES
    numeric = [f for f in context_features if f != "method"]

    print("[1/4] Clean-quality model...", flush=True)
    clean = build_clean_dataset(df)
    clean_train, clean_test = split_by_host(clean)
    clean_model, clean_metrics = evaluate_model(
        "clean_quality_grouped", make_pipeline(numeric, ["method"], MAIN_TREES, MAIN_LEAF),
        clean_train, clean_test, context_features, ["clean_psnr", "clean_ssim"],
    )
    joblib.dump(clean_model, MODEL_DIR / "clean_quality_model.joblib", compress=3)

    print("[2/4] Attack-recovery model...", flush=True)
    attacked = df[df["attack_type"] != "none"].dropna(subset=["attack_ber", "attack_correlation"]).copy()
    attack_features = context_features + ATTACK_FEATURES
    attack_train, attack_test = split_by_host(attacked)
    attack_model, attack_metrics = evaluate_model(
        "attack_recovery_grouped",
        make_pipeline([f for f in attack_features if f not in {"method", "attack_type"}],
                      ["method", "attack_type"], 200, 20),
        attack_train, attack_test, attack_features, ["attack_ber", "attack_correlation"],
    )
    joblib.dump(attack_model, MODEL_DIR / "attack_recovery_model.joblib", compress=3)
    del attacked, attack_train, attack_test

    print("[3/4] Strategy-performance model...", flush=True)
    strategy_data = build_strategy_robustness(df)
    strategy_train, strategy_test = split_by_host(strategy_data)
    strategy_model, strategy_metrics = evaluate_model(
        "strategy_performance_grouped", make_pipeline(numeric, ["method"], MAIN_TREES, MAIN_LEAF),
        strategy_train, strategy_test, context_features, OBJECTIVES,
    )
    joblib.dump(strategy_model, MODEL_DIR / "strategy_performance_model.joblib", compress=3)

    print("[4/4] Recommendation evaluation...", flush=True)
    summary, details, fixed = evaluate_recommendations(strategy_model, strategy_test, context_features, strategies)
    details.to_csv(OUTPUT_DIR / "recommendation_details.csv", index=False)
    fixed.to_csv(OUTPUT_DIR / "fixed_strategy_pareto_hit_rates.csv", index=False)

    all_metrics = pd.DataFrame(clean_metrics + attack_metrics + strategy_metrics)
    all_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    metadata = {
        "version": "five_method_v2",
        "random_seed": RANDOM_SEED,
        "split_rule": "GroupShuffleSplit by image_hash; 80/20 host-level split",
        "host_feature_count": len(HOST_FEATURES),
        "watermark_feature_count": len(WATERMARK_FEATURES),
        "context_feature_count": len(HOST_FEATURES) + len(WATERMARK_FEATURES),
        "forest_settings": {
            "clean_and_strategy": {"n_estimators": MAIN_TREES, "min_samples_leaf": MAIN_LEAF},
            "attack": {"n_estimators": 200, "min_samples_leaf": 20},
        },
        "clean_features": context_features,
        "attack_features": attack_features,
        "strategy_features": context_features,
        "strategy_targets": OBJECTIVES,
        "alpha_domain": alphas,
        "methods": methods,
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def r2(experiment: str, target: str) -> float:
        row = all_metrics[(all_metrics["experiment"] == experiment) & (all_metrics["target"] == target)]
        return float(row["R2"].iloc[0])

    model_summary = {
        "dataset_rows": int(len(df)),
        "hosts": int(df["image_hash"].nunique()),
        "hosts_by_dataset": df.groupby("dataset")["image_hash"].nunique().to_dict(),
        "methods": methods,
        "alphas": alphas,
        "strategies": len(strategies),
        "r2": {
            "clean_psnr": r2("clean_quality_grouped", "clean_psnr"),
            "clean_ssim": r2("clean_quality_grouped", "clean_ssim"),
            "clean_mse": r2("strategy_performance_grouped", "clean_mse"),
            "mean_attack_ber": r2("strategy_performance_grouped", "mean_attack_ber"),
            "embedding_time_seconds": r2("strategy_performance_grouped", "embedding_time_seconds"),
            "attack_ber": r2("attack_recovery_grouped", "attack_ber"),
            "attack_correlation": r2("attack_recovery_grouped", "attack_correlation"),
        },
        "recommendation": summary,
        "fixed_strategy_top5": fixed.head(5).to_dict(orient="records"),
        "forest_settings": metadata["forest_settings"],
    }
    (MODEL_DIR / "model_summary.json").write_text(json.dumps(model_summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print(all_metrics.to_string(index=False))
    print("\nRecommendation:")
    print(json.dumps(summary, indent=2, default=str))
    print("\nTop fixed strategies:")
    print(fixed.head(5).to_string(index=False))
    print(f"\nModels: {MODEL_DIR}")
    for path in sorted(MODEL_DIR.glob("*")):
        print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
