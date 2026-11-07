import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedKFold

from common import load, metric, OUT, SEED


def main():
    # --------------------------------------------------
    # 1. Load the original adjudicated dataset
    # --------------------------------------------------
    X, y, _, _ = load(policy="error")

    if len(y) != 184:
        raise ValueError(
            f"Expected 184 participants, found {len(y)}"
        )

    # --------------------------------------------------
    # 2. Load existing results
    # --------------------------------------------------
    original_metrics = pd.read_csv(
        OUT / "main_fold_metrics.csv"
    )

    # --------------------------------------------------
    # 3. Recreate the exact outer CV partitions
    # --------------------------------------------------
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=SEED
    )

    records = []

    for fold, (tr, te) in enumerate(cv.split(X, y)):

        y_train = y.iloc[tr]
        y_test = y.iloc[te]

        # IMPORTANT:
        # Median calculated using TRAINING data only.
        median_gpa = float(y_train.median())

        predictions = np.full(
            shape=len(te),
            fill_value=median_gpa,
            dtype=float
        )

        scores = metric(y_test, predictions)

        records.append({
            "tag": "median_baseline",
            "fold": fold,
            "repeat": fold // 5,
            "model": "Median",
            "train_n": len(tr),
            "test_n": len(te),
            "train_median_gpa": median_gpa,
            **scores
        })

    median_metrics = pd.DataFrame(records)

    # --------------------------------------------------
    # 4. Paired MAE comparisons
    # --------------------------------------------------
    paired = original_metrics.merge(
        median_metrics[["fold", "mae"]].rename(
            columns={"mae": "median_mae"}
        ),
        on="fold",
        validate="many_to_one"
    )

    # Negative difference = model performs better.
    paired["mae_difference"] = (
        paired["mae"] - paired["median_mae"]
    )

    paired_summary = (
        paired.groupby("model")
        .agg(
            model_mae=("mae", "mean"),
            median_mae=("median_mae", "mean"),
            mean_mae_difference=("mae_difference", "mean"),
            sd_mae_difference=("mae_difference", "std"),
            folds_better_than_median=(
                "mae_difference",
                lambda x: int((x < 0).sum())
            )
        )
        .reset_index()
        .sort_values("mean_mae_difference")
    )

    # --------------------------------------------------
    # 5. Median baseline summary
    # --------------------------------------------------
    median_summary = (
        median_metrics[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
    )

    # --------------------------------------------------
    # 6. Save results to outputs_v2
    # --------------------------------------------------
    median_metrics.to_csv(
        OUT / "median_baseline_fold_metrics.csv",
        index=False
    )

    median_summary.to_csv(
        OUT / "median_baseline_summary.csv"
    )

    paired.to_csv(
        OUT / "median_baseline_paired_by_fold.csv",
        index=False
    )

    paired_summary.to_csv(
        OUT / "median_baseline_paired_summary.csv",
        index=False
    )

    print("\nMedian baseline summary:")
    print(median_summary)

    print("\nPaired MAE comparisons:")
    print(paired_summary.to_string(index=False))

    print("\nResults saved to:", OUT)


if __name__ == "__main__":
    main()
