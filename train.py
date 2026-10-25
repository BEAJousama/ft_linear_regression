"""Train a one-feature linear regression model with batch gradient descent."""

# These modules help us read CSV, save JSON, check numbers, and locate files.
import csv
import json
import math
from pathlib import Path


# __file__ is this script; parent is the folder containing it.
DIRECTORY = Path(__file__).resolve().parent
# The / operator joins a folder and a filename when working with Path objects.
DATA_FILE = DIRECTORY / "data.csv"
MODEL_FILE = DIRECTORY / "model.json"
# The learning rate controls how big each training step is.
LEARNING_RATE = 0.1
# An underscore inside a number is just for readability: 10_000 means 10000.
ITERATIONS = 10_000


def load_data():
    # "with" closes the file automatically when we finish reading it.
    with DATA_FILE.open(newline="", encoding="utf-8") as file:
        # DictReader uses the CSV header, so a row can be accessed by column name.
        reader = csv.DictReader(file)
        # Make sure the file has both columns needed for training.
        if not reader.fieldnames or not {"km", "price"}.issubset(reader.fieldnames):
            # raise stops the function and reports a useful error.
            raise ValueError("data.csv must have km and price columns")

        # We will store each car as a pair: (mileage, price).
        rows = []
        # enumerate gives us both the row and its line number (line 1 is the header).
        for line_number, row in enumerate(reader, start=2):
            # CSV values are text, so convert them to decimal numbers.
            try:
                km = float(row["km"])
                price = float(row["price"])
            # If conversion fails, say which CSV line needs fixing.
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid number on line {line_number}") from error
            # Reject infinity, NaN ("not a number"), and negative mileage.
            if not math.isfinite(km) or not math.isfinite(price) or km < 0:
                raise ValueError(f"invalid value on line {line_number}")
            # append adds one (km, price) pair to the list.
            rows.append((km, price))

    # Training is impossible if the CSV contains no cars.
    if not rows:
        raise ValueError("data.csv has no examples")
    # Give the list of cars back to the code that called load_data().
    return rows


def train(rows):
    # Find the largest mileage. The _ means we do not need the price here.
    # Keep scale at least 1 so that we never divide by zero.
    scale = max(1.0, max(km for km, _ in rows))
    # Divide each mileage by scale; smaller numbers make training more stable.
    # This list still keeps each car's original price.
    examples = [(km / scale, price) for km, price in rows]
    # count is the number of cars, called "m" in the project formulas.
    count = len(examples)
    # Start with a prediction formula of 0 + 0 * mileage.
    theta0 = 0.0
    theta1 = 0.0

    # Repeat the learning step. _ means we do not use the loop number.
    for _ in range(ITERATIONS):
        # Reset the two gradient totals at the start of every round.
        gradient0 = 0.0
        gradient1 = 0.0
        # Use every car in this round: this is batch gradient descent.
        for mileage, price in examples:
            # Error = predicted price - real price. Mileage is scaled here.
            error = theta0 + theta1 * mileage - price
            # The first gradient adds each car's error.
            gradient0 += error
            # The second gradient weights the error by that car's mileage.
            gradient1 += error * mileage

        # Average the gradients, multiply by the learning rate, and subtract.
        # Store both answers before changing theta0 or theta1.
        next_theta0 = theta0 - LEARNING_RATE * gradient0 / count
        next_theta1 = theta1 - LEARNING_RATE * gradient1 / count
        # Assign both new values together, as the subject requires.
        theta0, theta1 = next_theta0, next_theta1

    # Training used scaled mileage. Convert theta1 back to price per real km.
    return theta0, theta1 / scale


def main():
    # Put file reading, training, and saving in try so errors print clearly.
    try:
        # Read (km, price) pairs from data.csv.
        rows = load_data()
        # Train returns the two numbers in the prediction formula.
        theta0, theta1 = train(rows)
        # Save them as JSON so predict.py can reuse them later.
        MODEL_FILE.write_text(
            # indent=2 makes the JSON readable; \n adds a final newline.
            json.dumps({"theta0": theta0, "theta1": theta1}, indent=2) + "\n",
            encoding="utf-8",
        )
    # OSError covers file problems; ValueError covers invalid data.
    except (OSError, ValueError) as error:
        raise SystemExit(f"Training failed: {error}") from error

    # Choose the sign to show between theta0 and the slope in the printed formula.
    sign = "+" if theta1 >= 0 else "-"
    # f-strings insert values inside {}; :.2f means two decimal places.
    print(f"Trained on {len(rows)} cars; saved model.json")
    print(f"price = {theta0:.2f} {sign} {abs(theta1):.8f} × km")


# Run main() only when this file is launched directly as a program.
if __name__ == "__main__":
    main()
