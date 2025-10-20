# Logistic Regression Examples

This repository contains a collection of Python scripts demonstrating various applications of logistic regression for binary classification tasks. The scripts use popular data science libraries such as `pandas`, `numpy`, `statsmodels`, `matplotlib`, and `seaborn`.

## Scripts

### `basic-logistic-regression.py`

This script provides a fundamental example of logistic regression. It uses the `Admittance.csv` dataset to predict student admittance based on their SAT scores. The script includes data loading, mapping categorical variables to numerical values, and plotting both a scatter plot of the data and the logistic regression curve.

### `binary-predictions-with-accuracy.py`

Building upon the basic example, this script demonstrates how to evaluate the accuracy of a logistic regression model. It uses the `Binary predictors.csv` dataset, which includes SAT scores and gender as predictors for admittance. The key feature of this script is the creation and interpretation of a confusion matrix to assess the model's performance.

### `bank-data-logistic-regression.py`

This script applies logistic regression to a real-world dataset, `Bank-data.csv`. The goal is to predict whether a customer will subscribe to a term deposit based on the duration of the marketing call. It covers data loading, preprocessing (including dropping unnecessary columns and mapping categorical variables), and a detailed interpretation of the logistic regression summary.

### `testing-the-model.py`

This script focuses on testing a trained logistic regression model on a separate dataset. It defines a function to calculate a confusion matrix and accuracy from a test dataset. It uses the `Test dataset.csv` to evaluate the model trained in `binary-predictions-with-accuracy.py`.

## Datasets

  * **`Admittance.csv`**: Contains two columns: `SAT` (student's SAT score) and `Admitted` (Yes/No).
  * **`Binary predictors.csv`**: Includes `SAT` scores, `Admitted` status (Yes/No), and `Gender` (Male/Female).
  * **`Bank-data.csv`**: A marketing dataset from a banking institution. The relevant columns for these examples are `duration` and `y` (subscription outcome).
  * **`Test dataset.csv`**: A smaller dataset with the same structure as `Binary predictors.csv`, used for testing the model.

## Requirements

To run these scripts, you will need to have the following Python libraries installed:

  * pandas
  * numpy
  * statsmodels
  * matplotlib
  * seaborn
  * scipy

You can install them using pip:

```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy
```

## How to Run

1.  Make sure you have all the required libraries installed.
2.  Place the Python scripts and the CSV files in the same directory.
3.  You can run each Python script from your terminal:

<!-- end list -->

```bash
python basic-logistic-regression.py
python binary-predictions-with-accuracy.py
python bank-data-logistic-regression.py
python testing-the-model.py
```
