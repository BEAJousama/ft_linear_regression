import json
import math
from pathlib import Path


MODEL_FILE = Path(__file__).resolve().parent / "model.json"


def load_model():
    if not MODEL_FILE.exists():
        return 0.0, 0.0

    try:
        model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
        theta0 = float(model["theta0"])
        theta1 = float(model["theta1"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError("model.json is invalid; run train.py again") from error
    if not math.isfinite(theta0) or not math.isfinite(theta1):
        raise ValueError("model.json is invalid; run train.py again")
    return theta0, theta1


def main():
    try:
        mileage = float(input("Mileage (km): "))
    except (EOFError, ValueError) as error:
        raise SystemExit("Prediction failed: enter a nonnegative number") from error
    if not math.isfinite(mileage) or mileage < 0:
        raise SystemExit("Prediction failed: enter a nonnegative number")

    try:
        theta0, theta1 = load_model()
    except ValueError as error:
        raise SystemExit(f"Prediction failed: {error}") from error

    estimate = theta0 + theta1 * mileage
    print(f"Estimated price: {estimate:.2f}")


if __name__ == "__main__":
    main()
