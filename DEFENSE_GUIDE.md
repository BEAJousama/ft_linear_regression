# Linear regression: theory, code, and evaluation guide

This guide explains the project in the order a peer evaluator is likely to ask about it. Read the commented `train.py` and `predict.py` alongside it. The [subject](en.subject.pdf) defines the required behavior; the [evaluation sheet](ft_linear_regression%20correction.pdf) lists the checks.

## The whole project in one minute

The CSV contains examples of cars, each with a mileage (`km`) and a price. Training finds two numbers, `theta0` and `theta1`, for this straight-line formula:

```text
estimated_price = theta0 + theta1 * mileage
```

`train.py` learns the two numbers and saves them in `model.json`. `predict.py` asks for a mileage, reads the saved numbers, and prints the estimate. The two files are separate programs. Before a model file exists, prediction uses `theta0 = 0` and `theta1 = 0`, so it prints `0.00` for any valid mileage.

```text
data.csv --read by--> train.py --writes--> model.json --read by--> predict.py
```

Use `python3 train.py` and then `python3 predict.py` from the project folder. To demonstrate prediction *before* training, use a fresh copy of the folder with no `model.json`.

## 1. The mathematical idea

### A straight line

The model has one input, mileage `x`, and one output, estimated price:

```text
h(x) = theta0 + theta1 * x
```

`theta0` is the intercept: the line's value when `x = 0`. `theta1` is its slope: the change in estimated price for one additional kilometer. A negative slope means the estimated price falls as mileage rises. The model cannot account for make, age, condition, or other features because they are not inputs.

### Error and objective

For car `i`, let `y_i` be its real price and `x_i` its mileage. Its prediction error is:

```text
error_i = h(x_i) - y_i
```

With `m` cars, training minimizes a squared-error objective:

```text
J(theta0, theta1) = (1 / (2m)) * sum(error_i ** 2)
```

Here `m = 24` for the supplied CSV. Squaring prevents positive and negative errors from canceling and gives large misses more weight. The `1/2` makes the derivative simpler; it does not change the best line.

### Gradients and gradient descent

A gradient tells us which way the error objective increases. The derivatives are:

```text
dJ/dtheta0 = (1/m) * sum(error_i)
dJ/dtheta1 = (1/m) * sum(error_i * x_i)
```

The second gradient includes `x_i` because changing the slope affects a prediction by `x_i` times as much as the same change to the intercept. Gradient descent subtracts a small multiple of each gradient:

```text
next_theta0 = theta0 - learning_rate * dJ/dtheta0
next_theta1 = theta1 - learning_rate * dJ/dtheta1
theta0, theta1 = next_theta0, next_theta1
```

Both `next_` values are calculated from the *same old* `theta0` and `theta1`, then assigned together. This is what the evaluation sheet calls simultaneous assignment. Each round uses **all** cars before updating the parameters, so this is batch gradient descent. The code repeats this for 10,000 rounds with a learning rate of `0.1`.

At the start, `theta0 = theta1 = 0`. For the first CSV car, `km = 240000` and `price = 3650`. Its initial prediction is `0`, so its error is `-3650`. That error contributes to both gradient totals; the program also adds the other 23 cars before taking the first step.

### Why the code scales mileage

Raw mileages are large, up to `240000`. In the slope gradient, each error is multiplied by mileage, making its numerical size very different from the intercept gradient. The code trains on:

```text
z = km / scale
scale = max(1, largest km)
```

For this CSV, `scale = 240000`, so training mileages are between 0 and 1. During training the line is `a + b*z`. Because `z = km/scale`, this equals `a + (b/scale)*km`. Therefore `train()` returns `theta0 = a` and `theta1 = b/scale`. The saved model and `predict.py` use **actual kilometers** and the exact required formula.

The scaling rule and learning rate work for this dataset; other datasets may need different settings. Scaling changes the internal units of the slope during training, not the form of the final prediction.

## 2. How the Python code implements it

### `train.py`: read, learn, save

| Code | What it means |
| --- | --- |
| `import csv`, `json`, `math`; `from pathlib import Path` | Use standard-library tools to parse CSV, save JSON, check numbers, and locate files. None fits a regression model. |
| `Path(__file__).resolve().parent` | Find the folder containing `train.py` so the CSV and model paths work from any launch directory. |
| `LEARNING_RATE`, `ITERATIONS` | Settings for the training loop. `10_000` is just Python's readable spelling of `10000`. |
| `load_data()` | Open `data.csv`, require `km` and `price` columns, convert each row to floats, reject invalid entries, and return a list of `(km, price)` pairs. |
| `for line_number, row in enumerate(reader, start=2)` | Visit each CSV row and remember its source line; line 1 is the header. |
| `rows.append((km, price))` | Add one tuple, or pair, to the training list. |
| `scale = max(1.0, max(km for km, _ in rows))` | Find the largest mileage while ignoring price (`_`); keep the divisor at least 1. |
| `examples = [(km / scale, price) for km, price in rows]` | Make a new list with scaled mileage and unchanged price. This is a list comprehension. |
| `theta0 = 0.0`; `theta1 = 0.0` | Start the learning parameters at zero. |
| Outer `for` loop | Repeat the learning step 10,000 times. The loop variable `_` is unused. |
| Inner `for mileage, price in examples` | Visit every car on every round; Python unpacks each pair into two names. |
| `error = theta0 + theta1 * mileage - price` | Compute estimated price minus real price. Here mileage is scaled. |
| `gradient0 += error`; `gradient1 += error * mileage` | Add this car's contribution to both gradients. `+=` means “add to the current value.” |
| `next_theta0`, `next_theta1` | Compute the two new values from the old parameters and average gradients. |
| `theta0, theta1 = next_theta0, next_theta1` | Update both parameters together after both new values exist. |
| `return theta0, theta1 / scale` | Convert the slope to price per real kilometer and return both numbers. |
| `main()` | Call `load_data()` and `train()`, write `model.json`, and print the learned equation. |
| `if __name__ == "__main__": main()` | Run `main()` when this file is launched directly. |

`try`/`except` catches expected bad input or file errors and shows a useful message. `with` closes the CSV file automatically. In `main()`, `json.dumps()` converts a Python dictionary to JSON text, while `write_text()` saves it. The saved keys are `theta0` and `theta1`.

### `predict.py`: ask, load, calculate

| Code | What it means |
| --- | --- |
| `MODEL_FILE = ... / "model.json"` | Find the saved model beside the script. |
| `if not MODEL_FILE.exists(): return 0.0, 0.0` | Use the required zero parameters before training. |
| `json.loads(MODEL_FILE.read_text(...))` | Read JSON text and turn it into a Python dictionary. |
| `float(model["theta0"])`, `float(model["theta1"])` | Take the saved parameter values by name and make sure they are numbers. |
| `math.isfinite(...)` | Reject infinity and NaN (“not a number”). |
| `float(input("Mileage (km): "))` | Ask the user for text and convert it to a number. |
| `estimate = theta0 + theta1 * mileage` | Apply the subject's prediction equation, using mileage in actual kilometers. |
| `print(f"Estimated price: {estimate:.2f}")` | Display the answer with two digits after the decimal point. |

`raise SystemExit(...)` stops the program with a readable error. An `f` before a quoted string lets Python insert values inside `{}`. Both scripts reject nonfinite numbers. `predict.py` also rejects negative mileage.

## 3. What the trained result means

The supplied 24 cars produce approximately:

```text
estimated_price = 8499.60 - 0.02144896 * km
```

The slope says the estimate drops by about `21.45` price units for every extra 1,000 km. For `100000` km, the predicted price is `6354.70`. The code does not name a currency because the CSV does not name one.

For the first CSV car (`240000` km, real price `3650`), the fitted line predicts `3351.85`, a gap of `298.15`. A gap is expected because one straight line cannot match every observed price exactly.

**Overfitting** means a model follows the training examples so closely that it performs worse on new examples. This model has only two learned parameters, so it has limited ability to overfit, but a small training error still does not prove generalization. To measure performance on new cars, evaluate on a separate set that was not used to train the model. The line may also produce unrealistic values far outside the training mileage range.

## 4. Evaluation sheet, item by item

These results were checked in a temporary copy so the real project folder remained in its pre-training state.

| Evaluation item | Evidence and result |
| --- | --- |
| Two programs | `train.py` and `predict.py` are separate and both run. **Pass.** |
| No library does regression | The imports are only `csv`, `json`, `math`, and `pathlib`. The gradient loop is written in `train.py`; there is no `numpy.polyfit`, scikit-learn regressor, or similar fitter. **Pass by code inspection.** |
| Prediction before learning | With no `model.json`, entering the nonzero mileage `100000` prints `Estimated price: 0.00`. **Pass.** |
| Required prediction equation | `predict.py` computes `theta0 + theta1 * mileage`. **Pass by code inspection and saved-model check.** |
| Learning and saving parameters | Training on the supplied CSV reports 24 cars and saves numeric `theta0` and `theta1` in `model.json`. **Pass.** |
| Reads the CSV as training reference | `load_data()` reads its rows; changing one CSV price by 1000 and retraining changed the learned intercept by about `-178.37`. **Pass.** |
| Simultaneous assignment | `next_theta0` and `next_theta1` are both computed before `theta0, theta1 = next_theta0, next_theta1`. **Pass by code inspection.** |
| Prediction after learning | Entering the known CSV mileage `240000` prints `3351.85`; the CSV price is `3650`. The estimate is coherent but not exact, as expected. **Pass.** |
| Repository submission | This local folder currently has no Git metadata, so a pushed submission cannot be verified here. Check your actual submission repository before defense. |
| Explain your own code | A program cannot prove this. Be ready to explain the formulas, scaling, and simultaneous assignment yourself. |

Only the mandatory part is implemented; the optional bonus items are outside this project scope.

### Why this is not a prohibited library solution

The prohibited shortcut is calling a function that already fits the line, such as `numpy.polyfit` or a regression estimator. This project uses only standard-library file and number utilities. The learning work is visible in the two nested loops of `train.py`: it calculates each prediction error, sums the gradients, and updates both parameters. `math.isfinite` only checks whether a number is valid; it does not learn coefficients.

### A short live defense sequence

1. In a fresh copy with no `model.json`, run `python3 predict.py` and enter `100000`. Explain why the answer is zero.
2. Run `python3 train.py`. Point to the CSV reader, error calculation, gradient totals, temporary values, and the saved JSON.
3. Run `python3 predict.py` again and enter `240000`. Compare `3351.85` with the CSV price `3650`.
4. Explain that scaling is undone before saving the slope, so prediction uses real kilometers.
5. Explain why the prediction need not exactly match a CSV price and what overfitting means.

The code and this guide were drafted with AI assistance. Review them and be able to explain each decision in your own words, as the subject requires.
