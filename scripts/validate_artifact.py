import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = PROJECT_ROOT / "reports/evaluation/project_validation.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


def get_dvc_status() -> dict:
    result = subprocess.run(
        ["uv", "run", "dvc", "status"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout.strip() or result.stderr.strip()

    clean = result.returncode == 0 and (not output or "up to date" in output.lower())

    return {
        "clean": clean,
        "return_code": result.returncode,
        "output": output,
    }


def main() -> None:
    required_files = {
        "metrics": (PROJECT_ROOT / "reports/evaluation/metrics.json"),
        "mlflow_run": (PROJECT_ROOT / "reports/evaluation/mlflow_run.json"),
        "registry_info": (PROJECT_ROOT / "reports/evaluation/registry_info.json"),
        "monitoring_summary": (
            PROJECT_ROOT / "reports/monitoring/monitoring_summary.json"
        ),
        "model": PROJECT_ROOT / "models/model.joblib",
        "dvc_pipeline": PROJECT_ROOT / "dvc.yaml",
        "dvc_lock": PROJECT_ROOT / "dvc.lock",
        "dockerfile": PROJECT_ROOT / "Dockerfile",
        "compose": PROJECT_ROOT / "compose.yaml",
        "readme": PROJECT_ROOT / "README.md",
        "architecture": (PROJECT_ROOT / "docs/architecture.md"),
        "evaluation_plan": (PROJECT_ROOT / "docs/evaluation-plan.md"),
    }

    file_status = {name: path.exists() for name, path in required_files.items()}

    metrics = load_json(required_files["metrics"])
    run_info = load_json(required_files["mlflow_run"])
    registry_info = load_json(required_files["registry_info"])
    monitoring = load_json(required_files["monitoring_summary"])
    dvc_status = get_dvc_status()

    required_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    checks = {
        "required_files_exist": all(file_status.values()),
        "metrics_complete": required_metrics.issubset(metrics),
        "mlflow_run_documented": all(
            key in run_info
            for key in {
                "run_id",
                "experiment_id",
                "model_uri",
                "tracking_uri",
            }
        ),
        "model_registered": (registry_info.get("status") == "registered"),
        "monitoring_expectations_met": monitoring.get(
            "all_expectations_met",
            False,
        ),
        "dvc_pipeline_up_to_date": dvc_status["clean"],
    }

    validation = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "required_files": file_status,
        "dvc_status": dvc_status,
        "model_metrics": metrics,
        "registry": {
            "status": registry_info.get("status"),
            "model_name": registry_info.get("registered_model_name"),
            "model_version": registry_info.get("model_version"),
            "model_alias": registry_info.get("model_alias"),
        },
        "monitoring": {
            "scenario_count": monitoring.get("scenario_count"),
            "successful_scenarios": monitoring.get("successful_scenarios"),
            "all_expectations_met": monitoring.get("all_expectations_met"),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(validation, file, indent=2)

    print(json.dumps(validation, indent=2))

    if not validation["all_checks_passed"]:
        raise SystemExit("Artifact validation failed.")


if __name__ == "__main__":
    main()
