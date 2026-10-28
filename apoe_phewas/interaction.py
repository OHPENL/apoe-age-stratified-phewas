"""Formal effect-modification models for selected PheCodes.

integrated_model  : one stacked model per PheCode with epsilon-4 x age stratum, x sex and
                    x race terms; person-clustered robust SEs account for individuals who
                    contribute to several strata (Reviewer 3 comment 2, Reviewer 4 comment 2).
continuous_age_model : epsilon-4 x natural spline of attained age in one record per person
                    (Reviewer 4, comment 1).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices, dmatrix
from scipy import stats

from .config import CONTINUOUS_AGE_GRID, MIN_CASES, RACES, SPLINE_DF
from .cohort import RACE_DUMMIES
from .models import MAX_ABS_BETA
from .phewas import outcome

Z975 = stats.norm.ppf(0.975)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0


def _joint_p(res, names):
    names = [n for n in names if n in res.params.index]
    if not names:
        return np.nan
    R = np.eye(len(res.params))[[res.params.index.get_loc(n) for n in names]]
    return float(res.wald_test(R, scalar=True).pvalue)


def integrated_model(windows: dict, counts_by_band: dict, phecode: str, pc_cols: list) -> dict:
    frames = []
    for k, band in enumerate(windows):
        w = windows[band]
        w = w[w['race'].isin(RACES)]
        c = counts_by_band[band]
        y, keep = outcome(w, c[c['phecode'] == phecode])
        frames.append(w.loc[keep].assign(y=y[keep], band=k).reset_index())
    D = pd.concat(frames, ignore_index=True)

    # Drop strata / race groups with too few cases to estimate an interaction term.
    band_cases = D.groupby('band')['y'].sum()
    D = D[D['band'].isin(band_cases[band_cases >= MIN_CASES].index)]
    race_cases = D.groupby('race')['y'].sum()
    D = D[D['race'].isin(race_cases[race_cases >= MIN_CASES].index)]
    bands = sorted(D['band'].unique())
    races = [r for r in RACES if r in set(D['race'])]
    if not bands or D['y'].sum() < MIN_CASES:
        return {'phecode': phecode, 'status': 'too_few_cases'}

    X = pd.DataFrame({'e4': D['e4'].astype(float)}, index=D.index)
    age_terms, race_terms = [], []
    for b in bands[1:]:
        X[f'band{b}'] = (D['band'] == b).astype(float)
        X[f'e4:band{b}'] = X['e4'] * X[f'band{b}']
        age_terms.append(f'e4:band{b}')
    X['male'] = D['male'].astype(float)
    X['e4:male'] = X['e4'] * X['male']
    for r in races[1:]:
        X[f'race_{r}'] = (D['race'] == r).astype(float)
        X[f'e4:{r}'] = X['e4'] * X[f'race_{r}']
        race_terms.append(f'e4:{r}')
    for col in ['age_window', 'fu_window'] + pc_cols:
        X[col] = _z(D[col].astype(float))
    X = sm.add_constant(X)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = sm.GLM(D['y'], X, family=sm.families.Binomial()).fit(
            cov_type='cluster', cov_kwds={'groups': pd.factorize(D['person_id'])[0]})

    out = {'phecode': phecode, 'status': 'ok', 'n_obs': len(D), 'n_persons': D['person_id'].nunique(),
           'n_cases': int(D['y'].sum()), 'bands_included': ';'.join(str(b) for b in bands),
           'races_included': ';'.join(races),
           'p_e4_x_age': _joint_p(res, age_terms), 'p_e4_x_sex': _joint_p(res, ['e4:male']),
           'p_e4_x_race': _joint_p(res, race_terms)}
    ci = res.conf_int()
    for term in ['e4'] + age_terms + ['e4:male'] + race_terms:
        out[f'ror_{term}'] = np.exp(res.params[term])        # ratio of odds ratios for interactions
        out[f'ror_{term}_lo'] = np.exp(ci.loc[term, 0])
        out[f'ror_{term}_hi'] = np.exp(ci.loc[term, 1])
    return out


def _lifetime_frame(cohort, lifetime: pd.DataFrame, phecode: str):
    """One record per person for a PheCode, restricted to the ages where cases occur
    (1st-99th percentile of case ages); outside that range the epsilon-4 x age terms are
    unidentified (no cases) and diverge. Returns (frame, lo, hi) or (status row, None, None)."""
    P = cohort.persons
    y, keep = outcome(P, lifetime[lifetime['phecode'] == phecode])
    d = P.loc[keep].copy()
    d['y'] = y[keep]
    d['age'] = d['last_age']
    if d['y'].sum() < MIN_CASES:
        return _status(phecode, 'too_few_cases', d), None, None
    lo, hi = d.loc[d['y'] == 1, 'age'].quantile([0.01, 0.99])
    d = d[(d['age'] >= lo) & (d['age'] <= hi)].copy()
    if d['y'].sum() < MIN_CASES:
        return _status(phecode, 'too_few_cases', d), None, None
    for col in ['followup_years'] + cohort.pc_cols:
        d[col] = _z(d[col].astype(float))
    return d, lo, hi


def _status(phecode, status, d):
    return pd.DataFrame([{'phecode': phecode, 'status': status, 'n_cases': int(d['y'].sum())}])


def _fit_pair(d, full_formula, reduced_rhs):
    """Full and reduced binomial GLMs on patsy design matrices (statsmodels moved the
    formula's design_info between versions, which broke the formula interface on some
    platforms). Returns (full, reduced, X_full), or None when the likelihood is not finite."""
    y_full, X_full = dmatrices(full_formula, d, return_type='dataframe')
    X_red = dmatrix(reduced_rhs, d, return_type='dataframe')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        full = sm.GLM(y_full, X_full, family=sm.families.Binomial()).fit()
        reduced = sm.GLM(y_full, X_red, family=sm.families.Binomial()).fit()
    for fit in (full, reduced):              # diverged coefficients can still give a finite llf
        e4_terms = [n for n in fit.params.index if 'e4' in n]
        if not (np.isfinite(fit.llf) and np.all(np.isfinite(fit.params))
                and np.all(np.isfinite(fit.bse)) and np.all(np.abs(fit.params[e4_terms]) < MAX_ABS_BETA)):
            return None
        # (spline and intercept coefficients may legitimately be large and offset each other)
    return full, reduced, X_full


def _adjustment(cohort, d):
    """Covariates for one PheCode. As in the main scan, a sex/race level in which every
    person is a control (or a case) separates the outcome, so that dummy is dropped."""
    keep = []
    for col in ['male'] + RACE_DUMMIES:
        level = d.loc[d[col] == 1, 'y']
        other = d.loc[d[col] == 0, 'y']
        if len(level) and len(other) and level.min() != level.max() and other.min() != other.max():
            keep.append(col)
    return ' + '.join(keep + ['followup_years'] + cohort.pc_cols)


def continuous_age_model(cohort, lifetime: pd.DataFrame, phecode: str,
                         grid=CONTINUOUS_AGE_GRID) -> pd.DataFrame:
    """OR per epsilon-4 allele as a smooth function of attained age, plus the LRT for interaction."""
    d, lo, hi = _lifetime_frame(cohort, lifetime, phecode)
    if lo is None:
        return d
    adj, spline = _adjustment(cohort, d), f'cr(age, df={SPLINE_DF})'
    fits = _fit_pair(d, f'y ~ e4 * {spline} + {adj}', f'e4 + {spline} + {adj}')
    if fits is None:
        return _status(phecode, 'not_converged', d)
    full, reduced, X_full = fits
    p_int = stats.chi2.sf(2 * (full.llf - reduced.llf), full.df_model - reduced.df_model)

    ages = [a for a in grid if lo <= a <= hi]
    base = {c: 0.0 for c in ['male', 'followup_years'] + RACE_DUMMIES + cohort.pc_cols}
    info = X_full.design_info
    X1 = np.asarray(build_design_matrices([info], pd.DataFrame({**base, 'age': ages, 'e4': 1.0}))[0])
    X0 = np.asarray(build_design_matrices([info], pd.DataFrame({**base, 'age': ages, 'e4': 0.0}))[0])
    diff = X1 - X0
    logor = diff @ full.params.to_numpy()
    se = np.sqrt(np.einsum('ij,jk,ik->i', diff, full.cov_params().to_numpy(), diff))
    return pd.DataFrame({'phecode': phecode, 'age': ages, 'or': np.exp(logor),
                         'or_lo': np.exp(logor - Z975 * se), 'or_hi': np.exp(logor + Z975 * se),
                         'p_e4_x_age_spline': p_int, 'n_cases': int(d['y'].sum()),
                         'age_min': lo, 'age_max': hi, 'status': 'ok'})


def linear_age_model(cohort, lifetime: pd.DataFrame, phecode: str) -> pd.DataFrame:
    """Sensitivity analysis: epsilon-4 x attained age as one linear term (age keeps its spline
    main effect). Reports the OR at age 60 and its multiplicative change per decade."""
    d, lo, hi = _lifetime_frame(cohort, lifetime, phecode)
    if lo is None:
        return d
    d['age10'] = (d['age'] - 60) / 10                 # centred at 60, per decade
    adj, spline = _adjustment(cohort, d), f'cr(age, df={SPLINE_DF})'
    fits = _fit_pair(d, f'y ~ e4 + e4:age10 + {spline} + {adj}', f'e4 + {spline} + {adj}')
    if fits is None:
        return _status(phecode, 'not_converged', d)
    full, reduced, _ = fits
    ci = full.conf_int()
    return pd.DataFrame([{
        'phecode': phecode, 'n_cases': int(d['y'].sum()), 'age_min': lo, 'age_max': hi,
        'or_at_60': np.exp(full.params['e4']), 'or_at_60_lo': np.exp(ci.loc['e4', 0]),
        'or_at_60_hi': np.exp(ci.loc['e4', 1]),
        'ratio_per_decade': np.exp(full.params['e4:age10']),
        'ratio_lo': np.exp(ci.loc['e4:age10', 0]), 'ratio_hi': np.exp(ci.loc['e4:age10', 1]),
        'p_e4_x_age_linear': stats.chi2.sf(2 * (full.llf - reduced.llf), 1), 'status': 'ok'}])
