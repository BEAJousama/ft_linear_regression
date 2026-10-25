"""Prompt for a car's mileage and estimate its price."""

# JSON reads the saved model; math checks numbers; Path locates the file.
import json
import math
from pathlib import Path


# Find model.json in the same folder as this script.
MODEL_FILE = Path(__file__).resolve().parent / "model.json"


def load_model():
    # Before training, the subject requires both parameters to be zero.
    if not MODEL_FILE.exists():
        return 0.0, 0.0

    # Try to read and understand the saved model.
    try:
        # read_text gets the file contents; json.loads turns JSON into a dict.
        model = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
        # Read both values by name and make sure they are numbers.
        theta0 = float(model["theta0"])
        theta1 = float(model["theta1"])
    # A missing, unreadable, or badly formatted model needs new training.
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError("model.json is invalid; run train.py again") from error
    # Also reject infinity and NaN ("not a number").
    if not math.isfinite(theta0) or not math.isfinite(theta1):
        raise ValueError("model.json is invalid; run train.py again")
    # Return both saved parameters to the caller.
    return theta0, theta1


def main():
    # input shows a prompt and returns text; float converts it to a number.
    try:
        mileage = float(input("Mileage (km): "))
    # A blank/bad answer, or no input, stops the program with a clear message.
    except (EOFError, ValueError) as error:
        raise SystemExit("Prediction failed: enter a nonnegative number") from error
    # Mileage must be a real, nonnegative number.
    if not math.isfinite(mileage) or mileage < 0:
        raise SystemExit("Prediction failed: enter a nonnegative number")

    # Load the parameters learned by train.py (or zeros before training).
    try:
        theta0, theta1 = load_model()
    # Show a useful message if model.json is invalid.
    except ValueError as error:
        raise SystemExit(f"Prediction failed: {error}") from error

    # This is the required prediction formula: theta0 + theta1 * mileage.
    estimate = theta0 + theta1 * mileage
    # :.2f displays the estimated price with two decimal places.
    print(f"Estimated price: {estimate:.2f}")


# Run main() only when this file is launched directly as a program.
if __name__ == "__main__":
    main()
