"""Phenome-wide scans within one age window and analysis group."""
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .config import EXCLUDE_SINGLE_MENTION, MIN_CASE_DATES, MIN_CASES, N_JOBS
from .models import fit_logistic

PARALLEL_MIN_TASKS = 40   # below this, a process pool costs more than it saves
_X = None                 # design matrix shared with worker processes
_BIN = []                 # columns of X holding binary covariates (sex / race dummies)


def design_matrix(df: pd.DataFrame, covariates: list) -> np.ndarray:
    """[1, e4, standardised covariates]; constant covariates (e.g. a race dummy inside one
    race group) are dropped. Standardising does not change the epsilon-4 odds ratio."""
    cov = [c for c in covariates if df[c].std() > 0]
    z = df[cov].astype(float)
    z = (z - z.mean()) / z.std()
    return np.column_stack([np.ones(len(df)), df['e4'].to_numpy(float), z.to_numpy()])


def outcome(df: pd.DataFrame, counts: pd.DataFrame):
    """y (1 = PheCode on >= MIN_CASE_DATES dates) and the rows kept for this PheCode."""
    n = counts.set_index('person_id')['n_dates'].reindex(df.index).fillna(0).to_numpy()
    y = (n >= MIN_CASE_DATES).astype(float)
    keep = ~((n > 0) & (n < MIN_CASE_DATES)) if EXCLUDE_SINGLE_MENTION else np.ones(len(n), bool)
    return y, keep


def add_fdr(res: pd.DataFrame, col: str = 'fdr') -> pd.DataFrame:
    res[col] = multipletests(res['p'], method='fdr_bh')[1] if len(res) else []
    return res


def binary_columns(X: np.ndarray) -> list:
    return [j for j in range(2, X.shape[1]) if len(np.unique(X[:, j])) == 2]


def _init_worker(X):
    global _X, _BIN
    _X, _BIN = X, binary_columns(X)
    # One BLAS thread per process avoids oversubscription. Where OMP_NUM_THREADS=1 is already
    # set before numpy loads (the Databricks notebook does this) threadpoolctl is not needed.
    if os.environ.get('OMP_NUM_THREADS') != '1':
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(1)
        except Exception:
            pass


def _fit_task(task):
    code, case_pos, drop_pos, method = task
    n = _X.shape[0]
    y = np.zeros(n)
    y[case_pos] = 1.0
    keep = np.ones(n, bool)
    keep[drop_pos] = False
    yk, Xk = y[keep], _X[keep]
    # A sex/race level with no cases (or only cases) separates the outcome and makes that
    # nuisance coefficient diverge; such columns are dropped for this PheCode only.
    drop = [j for j in _BIN if yk[Xk[:, j] == Xk[:, j].max()].min() == yk[Xk[:, j] == Xk[:, j].max()].max()]
    cols = [j for j in range(Xk.shape[1]) if j not in drop]
    try:
        fit = fit_logistic(yk, Xk[:, cols], method)
        fit['dropped_covariates'] = len(drop)
    except (np.linalg.LinAlgError, ValueError) as err:
        fit = {'converged': False, 'error': str(err)[:80]}
    return {'phecode': code, 'n_cases': len(case_pos),
            'n_controls': int(keep.sum()) - len(case_pos), **fit}


def scan(df: pd.DataFrame, covariates: list, counts: pd.DataFrame, phecodes=None,
         min_cases: int = MIN_CASES, method: str = 'auto', n_jobs: int = N_JOBS) -> pd.DataFrame:
    """Test every PheCode with >= min_cases cases among `df`; BH-FDR within this scan."""
    X = design_matrix(df, covariates)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    c = counts[counts['person_id'].isin(df.index)]
    if phecodes is not None:
        c = c[c['phecode'].isin(phecodes)]
    tasks = []
    for code, g in c.groupby('phecode', sort=True):
        is_case = g['n_dates'] >= MIN_CASE_DATES
        if is_case.sum() < min_cases:
            continue
        case_pos = pos[g.loc[is_case, 'person_id']].to_numpy()
        drop_pos = (pos[g.loc[~is_case, 'person_id']].to_numpy() if EXCLUDE_SINGLE_MENTION
                    else np.array([], int))
        tasks.append((code, case_pos, drop_pos, method))

    rows = None
    if n_jobs > 1 and len(tasks) >= PARALLEL_MIN_TASKS:
        for _attempt in range(2):          # a worker can die on Windows; retry with a fresh pool
            try:
                with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(X,)) as ex:
                    rows = list(ex.map(_fit_task, tasks, chunksize=4))
                break
            except (BrokenProcessPool, OSError) as err:
                print(f'    process pool failed ({type(err).__name__}); retrying')
    if rows is None:                       # serial (small scans, or the pool failed twice)
        _init_worker(X)
        rows = [_fit_task(t) for t in tasks]
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    ok = res['converged'] & res['p'].notna() if 'p' in res else res['converged'] & False
    return pd.concat([add_fdr(res[ok].copy()), res[~ok]], ignore_index=True)
