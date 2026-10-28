"""Logistic regression for the APOE epsilon-4 coefficient (column 1 of X).

Primary estimator: unpenalised maximum likelihood. If ML does not converge or shows
separation, the fit falls back to Firth's bias-reduced logistic regression, and Firth
is also run on every significant association as a sensitivity analysis (Reviewer 1,
comment 4; Reviewer 3, comment 6).
"""
import warnings

import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit

E4 = 1                 # column index of the epsilon-4 allele count in X
MAX_ABS_BETA = 15.0    # larger coefficients indicate (quasi-)separation
Z975 = stats.norm.ppf(0.975)


def _summary(beta, se, p, method, converged):
    lo, hi = beta - Z975 * se, beta + Z975 * se
    return {'beta': beta, 'se': se, 'p': p, 'or': np.exp(beta), 'or_lo': np.exp(lo),
            'or_hi': np.exp(hi), 'method': method, 'converged': bool(converged)}


def fit_mle(y, X):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = sm.Logit(y, X).fit(method='newton', maxiter=50, disp=0, warn_convergence=False)
    beta, se = res.params[E4], res.bse[E4]
    ok = (res.mle_retvals['converged'] and np.isfinite(se)
          and np.all(np.abs(res.params) < MAX_ABS_BETA))
    return _summary(beta, se, res.pvalues[E4], 'mle', ok)


def _firth(y, X, fixed=None, maxiter=100, tol=1e-8):
    """Firth-penalised Newton-Raphson; `fixed` constrains one coefficient to zero."""
    n, k = X.shape
    free = np.ones(k, bool)
    if fixed is not None:
        free[fixed] = False
    beta = np.zeros(k)

    def penalised(b):
        p = np.clip(expit(X @ b), 1e-12, 1 - 1e-12)
        w = p * (1 - p)
        info = (X * w[:, None]).T @ X
        _, logdet = np.linalg.slogdet(info)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * logdet
        return ll, p, w, info

    ll, p, w, info = penalised(beta)
    converged = False
    for _ in range(maxiter):
        h = w * np.einsum('ij,ji->i', X, np.linalg.solve(info, X.T))
        score = X.T @ (y - p + h * (0.5 - p))
        step = np.zeros(k)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        for _half in range(30):                      # step halving keeps the penalised LL rising
            new = penalised(beta + step)
            if new[0] >= ll - 1e-10:
                break
            step /= 2
        beta = beta + step
        ll, p, w, info = new
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    return beta, info, ll, converged


def fit_firth(y, X):
    """Firth estimate with a penalised likelihood-ratio p value and Wald confidence interval."""
    beta, info, ll, conv = _firth(y, X)
    _, _, ll0, conv0 = _firth(y, X, fixed=E4)
    se = np.sqrt(np.linalg.inv(info)[E4, E4])
    p = stats.chi2.sf(max(2 * (ll - ll0), 0.0), 1)
    return _summary(beta[E4], se, p, 'firth', conv and conv0)


def fit_logistic(y, X, method: str = 'auto'):
    """method: 'auto' (ML, Firth fallback) or 'firth'."""
    if method == 'firth':
        return fit_firth(y, X)
    try:
        res = fit_mle(y, X)
        if res['converged']:
            return res
    except (np.linalg.LinAlgError, ValueError):
        pass
    return fit_firth(y, X)
