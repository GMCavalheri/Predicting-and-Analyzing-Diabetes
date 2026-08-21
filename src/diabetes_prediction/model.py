"""Model factory: candidate classifiers compared during training."""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from diabetes_prediction.config import RANDOM_STATE


def get_models() -> dict[str, ClassifierMixin]:
    """Return the candidate models compared during training.

    `random_forest` uses 200 estimators, unlike the original notebook's
    `n_estimators=1` -- a single tree isn't actually a "forest", and its
    apparent performance came from the split/leakage bugs, not the model
    itself. `max_depth`/`min_samples_leaf` cap tree growth: left unbounded,
    trees on the weak-signal `minimal` feature set grew deep trying to fit
    noise, ballooning the persisted model to ~500MB with no accuracy
    benefit (measured) -- capping depth keeps the model a few MB and acts
    as regularization on top. `logistic_regression` is included as an
    interpretable baseline; `gradient_boosting` as a second, slower-but-
    often-stronger tree ensemble.
    """
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            criterion="entropy",
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
