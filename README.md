# Predicting credit card approvals

A logistic regression model that predicts whether a credit card application is approved.

This is a DataCamp guided project (2021). DataCamp wrote the task text and the code skeleton
(`# ... YOUR CODE FOR TASK n ...`); I filled in the code. I keep it here as early practice.

## Data

`Predicting Credit Card Approvals/datasets/cc_approvals.data`: the
[UCI Credit Approval dataset](https://archive.ics.uci.edu/dataset/27/credit+approval) (690 applications, 15 anonymised features plus the approved/denied label).
Missing values are marked with `?`.

## Approach

1. Replace `?` with NaN, fill numeric columns with the mean and text columns with the most frequent value.
2. Label-encode the text columns and drop features 11 and 13 (driver's licence and zip code).
3. Split 67/33 (random_state 42), scale with `MinMaxScaler`, and fit a `LogisticRegression`.
4. Grid-search `tol` and `max_iter` with 5-fold cross-validation.

## Results

| | Accuracy |
|---|---|
| Logistic regression, test set (228 applications) | 0.842 |
| Best 5-fold CV score in the grid search (`max_iter=100`, `tol=0.01`) | 0.854 |

## How to run

Open `Predicting Credit Card Approvals/notebook.ipynb` and run it from that folder.
The notebook imports `sklearn.grid_search`, which was removed in scikit-learn 0.20. On a current version,
use `from sklearn.model_selection import GridSearchCV` instead.

## Notes

Known issues, left as they were in 2021:

- The model is fit on the unscaled training set but scored on the scaled test set, and the scaler is fit again on the
  test set.
- The grid search runs on the whole dataset, so its score is not a held-out result.
