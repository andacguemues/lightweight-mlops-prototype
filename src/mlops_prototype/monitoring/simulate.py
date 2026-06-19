import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    """Load the central project configuration."""

    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_drifted_data(
    reference_data: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a deterministic dataset with deliberately shifted features."""

    drifted_data = reference_data.copy()
    rng = np.random.default_rng(random_state)

    changed_columns: list[str] = []

    numerical_changes = {
        "age": lambda values: (values + 15).clip(18, 100),
        "duration": lambda values: (values * 1.8 + 120).round(),
        "campaign": lambda values: values + 4,
        "emp.var.rate": lambda values: values + 2.0,
        "cons.price.idx": lambda values: values + 1.5,
        "cons.conf.idx": lambda values: values - 8.0,
        "euribor3m": lambda values: values + 1.2,
        "nr.employed": lambda values: values - 300.0,
    }

    for column, transformation in numerical_changes.items():
        if column in drifted_data.columns:
            drifted_data[column] = transformation(
                drifted_data[column]
            )
            changed_columns.append(column)

    categorical_changes = {
        "contact": "cellular",
        "month": "nov",
        "poutcome": "success",
        "job": "admin.",
    }

    for column, replacement in categorical_changes.items():
        if column not in drifted_data.columns:
            continue

        mask = rng.random(len(drifted_data)) < 0.8
        drifted_data.loc[mask, column] = replacement
        changed_columns.append(column)

    integer_columns = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
    ]

    for column in integer_columns:
        if column in drifted_data.columns:
            drifted_data[column] = (
                drifted_data[column]
                .round()
                .astype(int)
            )

    return drifted_data, changed_columns


def main() -> None:
    params = load_params()

    reference_path = (
        PROJECT_ROOT
        / params["monitoring"]["reference_path"]
    )
    no_drift_path = (
        PROJECT_ROOT
        / params["monitoring"]["no_drift_path"]
    )
    drifted_path = (
        PROJECT_ROOT
        / params["monitoring"]["drifted_path"]
    )
    manifest_path = (
        PROJECT_ROOT
        / params["monitoring"]["simulation_manifest_path"]
    )

    target = params["data"]["target"]
    random_state = params["monitoring"]["random_state"]

    reference_data = pd.read_csv(reference_path)

    if target in reference_data.columns:
        reference_features = reference_data.drop(
            columns=[target]
        )
    else:
        reference_features = reference_data.copy()

    no_drift_data = reference_features.copy()

    drifted_data, changed_columns = create_drifted_data(
        reference_data=reference_features,
        random_state=random_state,
    )

    no_drift_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    drifted_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    no_drift_data.to_csv(
        no_drift_path,
        index=False,
    )
    drifted_data.to_csv(
        drifted_path,
        index=False,
    )

    manifest = {
        "reference_path": str(
            reference_path.relative_to(PROJECT_ROOT)
        ),
        "no_drift_path": str(
            no_drift_path.relative_to(PROJECT_ROOT)
        ),
        "drifted_path": str(
            drifted_path.relative_to(PROJECT_ROOT)
        ),
        "row_count": len(reference_features),
        "feature_count": reference_features.shape[1],
        "random_state": random_state,
        "changed_columns": sorted(set(changed_columns)),
        "scenarios": {
            "no_drift": {
                "description": (
                    "Exact feature copy of the reference dataset."
                ),
                "expected_dataset_drift": False,
            },
            "drifted": {
                "description": (
                    "Numerical distributions and categorical "
                    "frequencies were deliberately shifted."
                ),
                "expected_dataset_drift": True,
            },
        },
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2)

    print(f"No-drift data saved to: {no_drift_path}")
    print(f"Drifted data saved to: {drifted_path}")
    print(f"Simulation manifest saved to: {manifest_path}")
    print(f"Changed columns: {sorted(set(changed_columns))}")


if __name__ == "__main__":
    main()