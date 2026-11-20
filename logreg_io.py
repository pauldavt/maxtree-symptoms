"""Save and load the logistic regression models as plain .npz files.

Only numbers are stored (scaler mean/scale, weights, intercept) plus JSON
metadata, so the files load with any numpy version, without pickle and
without scikit-learn.
"""

import json

import numpy as np


class LinearClassifier:
    """Standardisation followed by binary logistic regression (numpy only)."""

    def __init__(self, mean, scale, coef, intercept):
        self.mean, self.scale = mean, scale
        self.coef, self.intercept = coef, intercept

    @classmethod
    def from_sklearn(cls, pipeline):
        scaler, logreg = pipeline[0], pipeline[-1]
        if list(logreg.classes_) != [0, 1]:
            raise ValueError(f"Expected classes [0, 1], got {list(logreg.classes_)}")
        return cls(scaler.mean_, scaler.scale_, logreg.coef_[0], logreg.intercept_[0])

    def decision_function(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean) / self.scale) @ self.coef \
            + self.intercept

    def predict_proba(self, X):
        p1 = np.exp(-np.logaddexp(0.0, -self.decision_function(X)))  # sigmoid, no overflow
        return np.stack([1.0 - p1, p1], axis=1)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(np.int64)


def save_logreg(path, clf, meta, train_idx, val_idx):
    """Save a fitted sklearn pipeline (StandardScaler + LogisticRegression)."""
    m = LinearClassifier.from_sklearn(clf)
    arrays = {"mean": m.mean, "scale": m.scale, "coef": m.coef, "intercept": m.intercept}
    np.savez(path, **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()},
             train_idx=np.asarray(train_idx, dtype=np.int64),
             val_idx=np.asarray(val_idx, dtype=np.int64),
             meta=np.array(json.dumps(meta)))


def load_logreg(path):
    """Return (LinearClassifier, metadata dict)."""
    with np.load(path, allow_pickle=False) as f:
        model = LinearClassifier(f["mean"], f["scale"], f["coef"], f["intercept"][()])
        return model, json.loads(f["meta"][()])
