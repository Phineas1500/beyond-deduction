"""
Base class for linear probes using sklearn LogisticRegressionCV.

Per the MI research plan, logistic regression outperforms gradient-based
training on Gemma 2, and raw activations often outperform SAE features.
"""

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score


class BaseLinearProbe:
    """
    Base class for L2-regularized logistic regression probes.

    Uses sklearn's LogisticRegressionCV for automatic regularization tuning.
    """

    def __init__(
        self,
        n_folds: int = 5,
        Cs: int = 10,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str = 'balanced',
    ):
        """
        Initialize probe.

        Args:
            n_folds: Number of cross-validation folds
            Cs: Number of regularization values to try (log-spaced)
            max_iter: Maximum iterations for solver
            random_state: Random seed for reproducibility
            class_weight: Handle class imbalance ('balanced' recommended)
        """
        self.n_folds = n_folds
        self.Cs = Cs
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight

        self.model: Optional[LogisticRegressionCV] = None
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def _create_model(self) -> LogisticRegressionCV:
        """Create the sklearn model."""
        return LogisticRegressionCV(
            Cs=self.Cs,
            cv=StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state
            ),
            penalty='l2',
            solver='lbfgs',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=-1,  # Use all cores
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseLinearProbe':
        """
        Fit the probe.

        Args:
            X: Activations of shape (n_samples, hidden_size)
            y: Labels of shape (n_samples,)

        Returns:
            self
        """
        if len(X) == 0:
            raise ValueError("Cannot fit probe with empty data")

        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.model = self._create_model()
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Activations of shape (n_samples, hidden_size)

        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Activations of shape (n_samples, hidden_size)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before prediction")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score.

        Args:
            X: Activations
            y: True labels

        Returns:
            Accuracy
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before scoring")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self.model.score(X, y)

    def get_weights(self) -> np.ndarray:
        """
        Get probe weights for interpretability analysis.

        Returns:
            Weight matrix of shape (n_classes, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted to get weights")

        return self.model.coef_

    def get_bias(self) -> np.ndarray:
        """
        Get probe bias terms.

        Returns:
            Bias vector of shape (n_classes,)
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted to get bias")

        return self.model.intercept_

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Full evaluation with multiple metrics.

        Args:
            X: Activations
            y: True labels
            return_predictions: Whether to include predictions in output

        Returns:
            Dict with accuracy, balanced_accuracy, best_C, cv_scores, etc.
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before evaluation")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        result = {
            'accuracy': accuracy_score(y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'best_C': self.model.C_[0] if hasattr(self.model.C_, '__len__') else self.model.C_,
            'n_samples': len(y),
            'n_classes': len(self.classes_),
            'classes': self.classes_.tolist(),
            'classification_report': classification_report(y, y_pred, output_dict=True),
        }

        # Add CV scores if available
        if hasattr(self.model, 'scores_'):
            # scores_ is dict: class -> array of shape (n_folds, n_Cs)
            # We want the scores for the best C
            for class_label, scores in self.model.scores_.items():
                best_c_idx = np.argmax(scores.mean(axis=0))
                result[f'cv_mean_{class_label}'] = scores[:, best_c_idx].mean()
                result[f'cv_std_{class_label}'] = scores[:, best_c_idx].std()

        if return_predictions:
            result['predictions'] = y_pred
            result['probabilities'] = y_proba

        return result

    def save(self, path: str) -> None:
        """
        Save probe to disk.

        Args:
            path: File path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'BaseLinearProbe':
        """
        Load probe from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded probe instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


def train_and_evaluate_probe(
    probe_class,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    **probe_kwargs
) -> Tuple['BaseLinearProbe', Dict[str, Any]]:
    """
    Convenience function to train and evaluate a probe.

    Args:
        probe_class: Probe class to instantiate
        X: Activations of shape (n_samples, hidden_size)
        y: Labels of shape (n_samples,)
        test_size: Fraction of data for testing
        random_state: Random seed
        **probe_kwargs: Additional arguments for probe constructor

    Returns:
        (fitted_probe, evaluation_results)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Create and fit probe
    probe = probe_class(random_state=random_state, **probe_kwargs)
    probe.fit(X_train, y_train)

    # Evaluate
    results = probe.evaluate(X_test, y_test)
    results['n_train'] = len(X_train)
    results['n_test'] = len(X_test)

    return probe, results
