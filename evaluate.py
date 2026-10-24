"""Estimate the accuracy of the model on students it has never seen:
split the training dataset, train on one part, predict the other one.

Usage: python3 evaluate.py dataset_train.csv"""

import pandas as pd
import numpy as np
import argparse
import io
import contextlib
from utils.load_data import load_data
from utils.logreg import (ft_vec_mean, add_bias_column,
                          predict_probas, predict_houses)
from logreg_train import (FEATURE, get_houses,
                          fit_features, train_one_vs_all)
from logreg_predict import apply_features
from utils.stats import (ft_max, ft_min)


TRAIN_RATIO = 0.8
SEEDS = [0, 1, 2, 3, 4]
HYPERPARAMS = {
    "learning_rate": 0.4,
    "precision": 1e-7,
    "max_epochs": 10000,
}


def split_dataset(df: pd.DataFrame,
                  train_ratio=TRAIN_RATIO,
                  seed=SEEDS[0]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle the rows of df, then split them in two parts:
    the first `train_ratio` for the training, the rest for the validation.
    The same seed always gives the same split, to compare settings.
    Return (train, validation), both with an index reset from 0."""

    shuffled = df.sample(frac=1, random_state=seed)
    cut = int(len(shuffled) * train_ratio)
    train = shuffled.iloc[:cut]
    validation = shuffled.iloc[cut:]

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    return train, validation


def accuracy(predicted_houses, expected_houses) -> float:
    """Return the proportion of correct predictions, between 0 and 1.
    Accept lists, numpy arrays or pandas Series, of the same length
    and in the same student order."""

    predicted = np.asarray(predicted_houses)
    expected = np.asarray(expected_houses)
    return ft_vec_mean(predicted == expected)


def evaluate_split(train: pd.DataFrame, validation: pd.DataFrame,
                   features: list, hyperparams: dict,
                   batch_size: int) -> float:
    """Train the model on train with the given features, hyperparams
    and batch size (None: batch GD, 1: SGD, n: mini-batch), then
    predict validation, normalized with the train params only.
    The training logs are hidden.
    Return the accuracy on validation, between 0 and 1."""

    house_names, true_houses = get_houses(train)
    x_train_norm, norm_params = fit_features(train, features=features)
    X_train = add_bias_column(x_train_norm)
    with contextlib.redirect_stdout(io.StringIO()):
        theta_houses = train_one_vs_all(X_train, house_names, true_houses,
                                        batch_size, **hyperparams)
    model = {
        "features": features,
        "norm_params": norm_params,
        "thetas": theta_houses
    }

    x_valid_norm = apply_features(validation, model)
    X_valid = add_bias_column(x_valid_norm)
    probas_matrix, column_houses = predict_probas(X_valid, model)
    predicted_houses = predict_houses(probas_matrix, column_houses)

    return accuracy(predicted_houses, validation["Hogwarts House"])


def cross_validate(df: pd.DataFrame, features: list,
                   hyperparams: dict, batch_size: int,
                   seeds=SEEDS) -> list:
    """Repeat the split and the evaluation once per seed, so the score
    does not depend on one lucky or unlucky split.
    batch_size is given to the training (see evaluate_split).
    Return the list of accuracies, one per seed."""

    scores = []
    for seed in seeds:
        train, validation = split_dataset(df, train_ratio=TRAIN_RATIO,
                                          seed=seed)
        scores.append(evaluate_split(train, validation,
                                     features, hyperparams, batch_size))
    return scores


def main():
    """Main function that run the evaluation."""

    parser = argparse.ArgumentParser(
        description="Estimate the accuracy of the model by cross-validation"
                    " on the training dataset.",
        epilog="examples:\n"
               "  python3 evaluate.py dataset_train.csv\n"
               "  python3 evaluate.py dataset_train.csv --batch 32",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset", help="path to the training .csv file.")
    parser.add_argument("--batch", type=int, default=None,
                        metavar="SIZE",
                        help="number of students per update of theta:\n"
                             "  (none) batch gradient descent, all students"
                             " (default)\n"
                             "  1      stochastic gradient descent (SGD)\n"
                             "  n      mini-batch gradient descent,"
                             " n students")
    args = parser.parse_args()
    # ac = len(sys.argv)
    # if ac != 2:
    #     print("Error: the program must take one .csv file as argument.",
    #           file=sys.stderr)
    #     sys.exit(1)

    df = load_data(args.dataset)
    # features = [f for f in FEATURE if f != "Defense Against the Dark Arts"]
    # features = FEATURE + ["Arithmancy"] +
    # ["Astronomy"] + ["Potions"] + ["Care of Magical Creatures"]
    features = FEATURE
    scores = cross_validate(df, features, HYPERPARAMS, args.batch)
    for seed, score in zip(SEEDS, scores):
        print(f"seed {seed}: {score * 100:.2f}%")
    mean = ft_vec_mean(np.asarray(scores))
    print(f"Mean validation accuracy: {mean * 100:.2f}%"
          f" (min {ft_min(scores) * 100:.2f}%, max"
          f" {ft_max(scores) * 100:.2f}%)")


if __name__ == "__main__":
    main()
