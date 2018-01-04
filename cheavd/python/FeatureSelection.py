from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.svm import SVC


def lasso_select_features(X_train, y_train, X_test, alpha=1.0):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    model = SelectFromModel(lasso, prefit=True)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test


def l1norm_select_features(X_train, y_train, X_test, penalty=0.01):
    lsvc = LinearSVC(C=penalty, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test


def univ_select_features(X_train, y_train, X_test, n_fs):
    univ = SelectKBest(score_func=f_classif, k=n_fs)
    univ.fit(X_train, y_train)
    X_train = univ.transform(X_train)
    X_test = univ.transform(X_test)
    return X_train, X_test


def fdr_select_features(X_train, y_train, X_test, n_fs):
    fidx = []
    n_classes = np.max(y_train) + 1
    for i in range(n_classes):
        X_p = X_train[y_train == i]
        X_n = X_train[y_train != i]
        mu_p = np.mean(X_p, axis=0)
        mu_n = np.mean(X_n, axis=0)
        sigma_p = np.std(X_p, axis=0)
        sigma_n = np.std(X_n, axis=0)
        fdr = ((mu_p - mu_n)**2)/(sigma_p**2 + sigma_n**2)
        idx = np.argsort(-1 * fdr)
        fidx.append(idx[0:n_fs])
    fidx = np.asarray(fidx).reshape((n_classes*n_fs, ))
    fidx = np.unique(fidx)
    return X_train[:, fidx], X_test[:, fidx]


def rfe_select_features(X_train, y_train, X_test, n_fs, penalty=1, step=1):
    svc = SVC(kernel="linear", C=penalty)
    rfe = RFE(estimator=svc, n_features_to_select=n_fs, step=step)
    rfe.fit(X_train, y_train)
    mask = rfe.support_
    fidx = np.where(mask)[0].tolist()
    X_train = X_train[:, fidx]
    X_test = X_test[:, fidx]
    return X_train, X_test