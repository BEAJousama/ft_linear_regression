import csv
import json
import math
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
DATA_FILE = DIRECTORY / "data.csv"
MODEL_FILE = DIRECTORY / "model.json"
LEARNING_RATE = 0.1
ITERATIONS = 10_000


def load_data():
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"km", "price"}.issubset(reader.fieldnames):
            raise ValueError("data.csv must have km and price columns")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                km = float(row["km"])
                price = float(row["price"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid number on line {line_number}") from error
            if not math.isfinite(km) or not math.isfinite(price) or km < 0:
                raise ValueError(f"invalid value on line {line_number}")
            rows.append((km, price))

    if not rows:
        raise ValueError("data.csv has no examples")
    return rows


def train(rows):
    scale = max(1.0, max(km for km, _ in rows))
    examples = [(km / scale, price) for km, price in rows]
    count = len(examples)
    theta0 = 0.0
    theta1 = 0.0

    for _ in range(ITERATIONS):
        gradient0 = 0.0
        gradient1 = 0.0
        for mileage, price in examples:
            error = theta0 + theta1 * mileage - price
            gradient0 += error
            gradient1 += error * mileage

        next_theta0 = theta0 - LEARNING_RATE * gradient0 / count
        next_theta1 = theta1 - LEARNING_RATE * gradient1 / count
        theta0, theta1 = next_theta0, next_theta1

    return theta0, theta1 / scale


def main():
    try:
        rows = load_data()
        theta0, theta1 = train(rows)
        MODEL_FILE.write_text(
            json.dumps({"theta0": theta0, "theta1": theta1}, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error

    sign = "+" if theta1 >= 0 else "-"
    print(f"Trained on {len(rows)} cars; saved model.json")
    print(f"price = {theta0:.2f} {sign} {abs(theta1):.8f} × km")


if __name__ == "__main__":
    main()
