from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    OUTPUT_DIR,
    PROCESSED_DIR,
    display_path,
    precision_at_k,
    write_json,
)


FEATURE_PATH = PROCESSED_DIR / "refresh_feature_vector.csv"
BASELINE_PATH = PROCESSED_DIR / "baseline_refresh_queue.csv"
PREDICTION_PATH = PROCESSED_DIR / "model_predictions.csv"
RESULT_PATH = OUTPUT_DIR / "model_results.json"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train refresh opportunity models.")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--predictions", default=str(PREDICTION_PATH))
    parser.add_argument("--results", default=str(RESULT_PATH))
    return parser.parse_args()


def build_feature_matrix(
    frame: pd.DataFrame,
    *,
    fit_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build numeric/one-hot features; learn categorical columns from fit rows only."""
    numeric_features = [
        column for column in MODEL_NUMERIC_FEATURES if column in frame.columns
    ]
    categorical_features = [
        column for column in MODEL_CATEGORICAL_FEATURES if column in frame.columns
    ]

    numeric_frame = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan).fillna(0)

    categorical_frame = frame[categorical_features].fillna("unknown").astype(str)
    encoded_frame = pd.get_dummies(
        categorical_frame,
        prefix=categorical_features,
        dummy_na=False,
        dtype=float,
    )
    if fit_indices is not None and categorical_features:
        fit_encoded = pd.get_dummies(
            categorical_frame.iloc[fit_indices],
            prefix=categorical_features,
            dummy_na=False,
            dtype=float,
        )
        # Categories only found in validation/test are unknown to the fitted
        # feature schema and map to an all-zero encoded representation.
        encoded_frame = encoded_frame.reindex(columns=fit_encoded.columns, fill_value=0)

    feature_frame = pd.concat(
        [numeric_frame.reset_index(drop=True), encoded_frame.reset_index(drop=True)],
        axis=1,
    )
    return feature_frame, list(feature_frame.columns)


def make_client_aware_split(
    frame: pd.DataFrame,
    target_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, str]:
    all_indices = np.arange(len(frame))
    client_series = frame["client_id"].fillna("unknown").astype(str)
    unique_clients = client_series.drop_duplicates().to_numpy()

    if len(unique_clients) >= 5:
        random_generator = np.random.default_rng(RANDOM_STATE)
        shuffled_clients = random_generator.permutation(unique_clients)
        test_client_count = max(1, int(round(len(shuffled_clients) * 0.2)))
        test_clients = set(shuffled_clients[:test_client_count])
        test_mask = client_series.isin(test_clients).to_numpy()
        train_indices = all_indices[~test_mask]
        test_indices = all_indices[test_mask]

        if (
            len(train_indices) > 0
            and len(test_indices) > 0
            and target_series.iloc[train_indices].nunique() == 2
            and target_series.iloc[test_indices].nunique() == 2
        ):
            return train_indices, test_indices, "client_holdout"

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target_series,
    )
    return np.array(train_indices), np.array(test_indices), "stratified_row_holdout"


def make_validation_split(
    outer_train_indices: np.ndarray,
    target_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Split outer training rows for model selection without touching the test set."""
    outer_target = target_series.iloc[outer_train_indices]
    class_counts = outer_target.value_counts()
    if outer_target.nunique() != 2 or class_counts.min() < 2:
        raise ValueError(
            "The outer training partition needs at least two examples of each "
            "target class to create a separate validation split."
        )
    fit_indices, validation_indices = train_test_split(
        outer_train_indices,
        test_size=0.2,
        random_state=43,
        stratify=outer_target,
    )
    return np.asarray(fit_indices), np.asarray(validation_indices)


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=10,
            min_samples_leaf=25,
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def predict_probability(model: object, feature_frame: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model does not expose predict_proba: {type(model)!r}")
    probabilities = model.predict_proba(feature_frame)
    return np.asarray(probabilities[:, 1], dtype=float)


def metric_payload(
    target_series: pd.Series,
    probability_scores: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    binary_predictions = (probability_scores >= 0.5).astype(int)
    payload = {
        f"{prefix}accuracy": float(accuracy_score(target_series, binary_predictions)),
        f"{prefix}precision": float(
            precision_score(target_series, binary_predictions, zero_division=0)
        ),
        f"{prefix}recall": float(
            recall_score(target_series, binary_predictions, zero_division=0)
        ),
        f"{prefix}f1": float(f1_score(target_series, binary_predictions, zero_division=0)),
        f"{prefix}precision_at_20": precision_at_k(target_series, probability_scores, 20),
        f"{prefix}precision_at_50": precision_at_k(target_series, probability_scores, 50),
        f"{prefix}precision_at_100": precision_at_k(target_series, probability_scores, 100),
    }
    if target_series.nunique() == 2:
        payload[f"{prefix}roc_auc"] = float(roc_auc_score(target_series, probability_scores))
        payload[f"{prefix}average_precision"] = float(
            average_precision_score(target_series, probability_scores)
        )
    else:
        payload[f"{prefix}roc_auc"] = 0.0
        payload[f"{prefix}average_precision"] = 0.0
    return payload


def top_feature_importance(
    model: object,
    feature_columns: list[str],
    *,
    limit: int = 25,
) -> list[dict[str, float | str]]:
    if isinstance(model, Pipeline):
        classifier = model.named_steps["model"]
        raw_values = np.abs(classifier.coef_[0])
    elif hasattr(model, "feature_importances_"):
        raw_values = np.asarray(model.feature_importances_, dtype=float)
    else:
        raw_values = np.zeros(len(feature_columns), dtype=float)

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": raw_values,
        }
    )
    importance_frame = importance_frame.sort_values("importance", ascending=False).head(limit)
    return [
        {"feature": str(row.feature), "importance": float(row.importance)}
        for row in importance_frame.itertuples(index=False)
    ]


def main() -> None:
    args = parse_args()

    frame = pd.read_csv(args.features)
    baseline_frame = pd.read_csv(args.baseline)
    if frame.empty:
        raise ValueError("Feature vector is empty")
    if "is_declining_label" not in frame.columns or "client_id" not in frame.columns:
        raise ValueError("Feature vector must include is_declining_label and client_id")
    target_series = pd.to_numeric(frame["is_declining_label"], errors="raise").astype(int)
    if target_series.nunique() != 2:
        raise ValueError("Training target has only one class; cannot train classifier")

    train_indices, test_indices, split_strategy = make_client_aware_split(frame, target_series)
    fit_indices, validation_indices = make_validation_split(train_indices, target_series)

    # The validation and test rows do not influence the encoded categorical
    # vocabulary or model fit used for model selection/evaluation.
    evaluation_features, evaluation_columns = build_feature_matrix(frame, fit_indices=fit_indices)
    fit_features = evaluation_features.iloc[fit_indices]
    validation_features = evaluation_features.iloc[validation_indices]
    test_features = evaluation_features.iloc[test_indices]
    fit_target = target_series.iloc[fit_indices]
    validation_target = target_series.iloc[validation_indices]
    test_target = target_series.iloc[test_indices]

    baseline_lookup = baseline_frame.set_index("content_id")["baseline_refresh_score"]
    validation_baseline_scores = (
        frame.iloc[validation_indices]["content_id"].map(baseline_lookup).fillna(0).to_numpy()
    )
    test_baseline_scores = (
        frame.iloc[test_indices]["content_id"].map(baseline_lookup).fillna(0).to_numpy()
    )
    baseline_validation_metrics = metric_payload(
        validation_target,
        validation_baseline_scores,
        prefix="baseline_",
    )
    baseline_test_metrics = metric_payload(
        test_target,
        test_baseline_scores,
        prefix="baseline_test_",
    )

    validation_results: dict[str, dict[str, float]] = {}
    for model_name, model in build_models().items():
        model.fit(fit_features, fit_target)
        validation_probabilities = predict_probability(model, validation_features)
        validation_results[model_name] = metric_payload(
            validation_target,
            validation_probabilities,
        )

    # Select only from validation metrics. Never rank/select models on test metrics.
    best_model_name = sorted(
        validation_results,
        key=lambda name: (
            validation_results[name]["precision_at_50"],
            validation_results[name]["average_precision"],
            validation_results[name]["roc_auc"],
        ),
        reverse=True,
    )[0]

    # Evaluate the selected candidate once on the untouched outer test partition.
    selected_eval_model = build_models()[best_model_name]
    selected_eval_model.fit(evaluation_features.iloc[train_indices], target_series.iloc[train_indices])
    selected_test_probabilities = predict_probability(selected_eval_model, test_features)
    selected_test_metrics = metric_payload(
        test_target,
        selected_test_probabilities,
        prefix="test_",
    )

    # Refit all candidates on all available rows for the operational refresh
    # ranking output. These scores are ranking signals, not held-out evaluations.
    full_features, full_feature_columns = build_feature_matrix(frame)
    full_data_models = build_models()
    prediction_frame = frame[["content_id", "client_id"]].copy()
    for model_name, model in full_data_models.items():
        model.fit(full_features, target_series)
        prediction_frame[f"prob_{model_name}"] = predict_probability(model, full_features)

    prediction_frame["best_model_name"] = best_model_name
    prediction_frame["best_model_probability"] = prediction_frame[f"prob_{best_model_name}"]

    prediction_path = Path(args.predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)

    best_full_model = full_data_models[best_model_name]
    results_payload = {
        "input_rows": int(len(frame)),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "test_rows": int(len(test_indices)),
        "split_strategy": split_strategy,
        "model_selection_partition": "inner_validation",
        "test_partition_used_for_selection": False,
        "target": "is_declining_label",
        "target_positive_rows": int(target_series.sum()),
        "target_positive_rate": float(target_series.mean()),
        "feature_count": int(len(full_feature_columns)),
        "model_numeric_features": MODEL_NUMERIC_FEATURES,
        "model_categorical_features": MODEL_CATEGORICAL_FEATURES,
        "models": validation_results,
        "baseline": baseline_validation_metrics,
        "held_out_test": {
            "selected_model": best_model_name,
            "selected_model_metrics": selected_test_metrics,
            "baseline_metrics": baseline_test_metrics,
        },
        "best_model": {
            "name": best_model_name,
            "selection_metric": "validation_precision_at_50",
            "validation_metrics": validation_results[best_model_name],
            "test_metrics": selected_test_metrics,
            "feature_importance_top": top_feature_importance(best_full_model, full_feature_columns),
        },
        "prediction_output": display_path(prediction_path),
        "prediction_note": "Probabilities in model_predictions.csv are generated by models refit on all rows for prioritization; do not interpret them as held-out test predictions.",
    }
    write_json(Path(args.results), results_payload)

    print(f"Trained {len(validation_results)} models on {len(frame):,} rows")
    print(f"Split strategy: {split_strategy}")
    print(f"Selected using validation precision@50: {best_model_name}")
    print(f"Selected-model held-out test precision@50: {selected_test_metrics['test_precision_at_50']:.4f}")
    print(f"Wrote predictions: {prediction_path}")
    print(f"Wrote model results: {args.results}")


if __name__ == "__main__":
    main()
