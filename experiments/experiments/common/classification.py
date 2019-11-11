from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fda_methods.rk import RK
import matplotlib.pyplot as plt


class PLS(PLSRegression):

    def transform(self, X, Y=None, copy=True):
        return PLSRegression.predict(self, X, copy)


def fdatagrid_with_resolution(array, resolution):
    step = (array.data_matrix.shape[1] - 1) // 2**resolution

    return array[:, ::step]


def dict_with_resolution(d, resolution):
    d_new = {}

    for key, value in d.items():
        d_new[key] = fdatagrid_with_resolution(value, resolution)

    return d_new


def plot_with_var(mean, std, color, label, std_span=0, **kwargs):

    for multiple in range(std_span, 0, -1):
        plt.fill_between(range(len(mean)), mean - multiple *
                         std, mean + multiple * std, color=color, alpha=0.15)
    plt.plot(mean, label=label, color=color, **kwargs)


def classifier_lda(n_features, cv):
    return LinearDiscriminantAnalysis(priors=[.5, .5])


def classifier_qda(n_features, cv):
    return QuadraticDiscriminantAnalysis(priors=[.5, .5])


def classifier_pls_centroid(n_features, cv):
    return GridSearchCV(Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLS()),
        ("centroid", NearestCentroid())]),
        param_grid={
        "pls__n_components": range(1, min(21, n_features))
    }, cv=cv)


def classifier_pca_qda(n_features, cv):
    return GridSearchCV(Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=0)),
        ("qda", QuadraticDiscriminantAnalysis())]),
        param_grid={
        "pca__n_components": range(1, min(21, n_features))
    }, cv=cv)


def classifier_rkc(n_features, cv):
    return GridSearchCV(Pipeline([
        ("rk", RK()),
        ("lda", LinearDiscriminantAnalysis())]),
        param_grid={
        "rk__n_components": range(1, min(21, n_features))
    }, cv=cv)
