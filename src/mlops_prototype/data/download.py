from pathlib import Path
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAIN_ZIP_PATH = RAW_DIR / "bank_marketing.zip"
NESTED_ZIP_PATH = RAW_DIR / "bank-additional.zip"
TARGET_FILE = RAW_DIR / "bank-additional" / "bank-additional-full.csv"

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists():
        print(f"Dataset already exists: {TARGET_FILE}")
        return

    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, MAIN_ZIP_PATH)

    print("Extracting main archive...")
    with zipfile.ZipFile(MAIN_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print("Extracting nested archive...")
    with zipfile.ZipFile(NESTED_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Expected file not found: {TARGET_FILE}")

    print(f"Dataset available at: {TARGET_FILE}")


if __name__ == "__main__":
    main()
