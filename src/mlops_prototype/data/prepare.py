from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    params = load_params()

    raw_path = PROJECT_ROOT / params["data"]["raw_path"]
    train_path = PROJECT_ROOT / params["data"]["train_path"]
    test_path = PROJECT_ROOT / params["data"]["test_path"]
    target = params["data"]["target"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    df = pd.read_csv(raw_path, sep=";")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target],
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()