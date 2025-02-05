import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y

"""
    Original Author: Andreas Kempa-Liehr
    Last updated: 31.10.2019
    This is a clone/copy of the bayesian functions from DSLab 
"""


def rms(w, x, t):
    """Returns root mean square error of a polynomial model.

    Args:
        w (iterable): Polynomial coefficients.
        x (iterable): Samples.
        t (iterable): Target values with same shape as x.

    Returns:
        float: Root mean square error.
    """
    y = np.poly1d(w)(x)
    loss = 0.5 * np.sum((y - t) ** 2)
    return np.sqrt(2 * loss / x.size)


class BayesianPolynomialRegression(BaseEstimator, RegressorMixin):
    """A Bayesian linear regression with Gaussian prior.

    From Kempa-Liehr (2016): Don't Overfit - The Engineers' (totally
    incomplete) Guide to Computational Analytics.

    Parameters:
        M (int, optional): Complexity with respect to order of
            polynomials.
        alpha (float, optional): Regularization parameter.
    """

    def __init__(self, M=10, alpha=1e-3):
        self.M = M  # Complexity with respect to order of polynomials
        self.alpha = alpha  # Regularization parameter

    def fit(self, X, y):
        """Fitting function for Bayesian linear regression with Gaussian prior.

        Args:
            X (array-like or sparse matrix): The training input samples.
                Of shape [n_samples, 1].
            y (array-like): The target values.
                Class labels in classification real numbers in regression.
                Of shape [n_samples].

        Returns:
            object: Returns self.
        """
        X, y = check_X_y(X, y)

        xs = X[:, 0]
        ts = y

        M = self.M
        b_i = np.vectorize(lambda i: np.sum(ts * xs ** i))
        A_ij = np.vectorize(lambda i, j: np.sum(xs ** (i + j)))
        b = np.fromfunction(b_i, (M + 1,))
        A = np.fromfunction(A_ij, (M + 1, M + 1))
        S = lambda alpha: A + alpha * np.identity(M + 1)
        w_reg = np.vectorize(lambda alpha: np.linalg.solve(S(alpha), b)[::-1])

        # Estimates mean
        alpha = self.alpha
        m = np.vectorize(lambda x: np.poly1d(w_reg(alpha))(x))
        self.m = m

        # Computes optimal precision
        beta_ML = rms(w_reg(alpha), xs, ts) ** (-2)

        phi = lambda x: x ** np.arange(M + 1)
        # Estimate variance
        S_inv = np.linalg.inv(S(alpha))
        s2 = np.vectorize(lambda x: 1. / beta_ML * (1 + phi(x).transpose().dot(
            S_inv).dot(phi(x))))
        self.s2 = s2

        # Returns the estimator
        return self

    def predict_cdf(self, X):
        """Estimates conditional density functions.

        Args:
            X (array-like of shape = [n_samples, n_features]):
                The input samples.

        Returns:
            y (array of shape = [n_samples]):
                Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        m = self.m
        s2 = self.s2
        x = X[:, 0]
        polyfit_cdf = lambda x: norm(loc=m(x), scale=np.sqrt(s2(x)))

        try:
            result = [polyfit_cdf(a) for a in x]
        except TypeError:  # x is not iterable
            result = [polyfit_cdf(x)]
        return result

    def predict(self, X):
        """Predict function for estimator.

        Args:
            X (array-like of shape = [n_samples, n_features]):
                The input samples.

        Returns:
            y (array of shape = [n_samples])
                Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        cdfs = self.predict_cdf(X)
        result = np.array(list(map(lambda pdf: pdf.mean(), cdfs)))
        return result


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    """A Bayesian linear regression with Gaussian prior.

    Differs from BayesianPolynomialRegression in that the feature
    matrix must be passed explicitly.

    From Kempa-Liehr (2016): Don't Overfit - The Engineers' (totally
    incomplete) Guide to Computational Analytics.

    Parameters:
        alpha (float, optional): Regularization parameter.
        fit_intercept (bool, optional): Whether to calculate with an
            intercept.
    """

    def __init__(self, alpha=1e-3, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fitting function for Bayesian linear regression with Gaussian prior.

        Args:
            X (array-like or sparse matrix): The training input feature matrix.
                Of shape [n_samples, n_feats]).
            y (array-like): The target values.
                Targets are class labels in classification, real numbers
                in regression. Of shape [n_samples].

        Returns:
            object: Returns self.
        """
        X, y = check_X_y(X, y)

        # Compute optimal weights for regularised linear regression
        # Andreas does polynomial expansion; we'll do that outside this code
        # Implement eqn (3.28) in Bishop, with phi = X
        # NOTE: Here, we treat Bishop's lambda and alpha as the same
        alpha = self.alpha
        N, M = np.shape(X)

        w_reg = np.dot(
            np.dot(np.linalg.inv(alpha * np.identity(M) + np.dot(X.T, X)),
                   X.T), y)

        self.w_reg = w_reg

        # Computes optimal precision
        self.beta_ML = 1 / (np.sum((y - np.sum(w_reg * X, axis=1)) ** 2) / N)

        # Matrix S, according to Bishop's definition.
        # Used for mean and variance calculations.
        S_inv = alpha * np.identity(M) + self.beta_ML * np.dot(X.T, X)
        S = np.linalg.inv(S_inv)
        self.S = S

        return self

    def predict_cdf(self, X):
        """Estimates conditional density functions.

        Args:
            X (array-like): The input samples.
                Of shape [n_samples, n_features].

        Returns:
            y (array): Returns \\[x^2]\\ where \\[x]\\ is the first column
                of \\[X]\\. Of shape = [n_samples].
        """
        w_reg = self.w_reg
        beta_ML = self.beta_ML
        S = self.S

        # Mean and variance
        m = np.dot(X, w_reg)
        s2 = beta_ML ** (-1) + np.diagonal(np.dot(np.dot(X, S), X.T))

        # Fit each sample to a normal dist
        num_samples = X.shape[0]
        normals = [
            norm(loc=m[i], scale=np.sqrt(s2[i]))
            for i in np.arange(0, num_samples)
        ]
        return normals

    def predict(self, X):
        """Predict function for estimator.

        Args:
            X (array-like of shape = [n_samples, n_features]):
                The input samples.

        Returns:
            y (array of shape = [n_samples])
                Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        cdfs = self.predict_cdf(X)
        result = np.array(map(lambda pdf: pdf.mean(), cdfs))
        return result
