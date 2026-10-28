"""Table 1 and ancestry-PC summaries (Reviewer 1, comments 3 and 6)."""
import numpy as np
import pandas as pd
from scipy import stats

from .config import AGE_BANDS


def _band_label(age: pd.Series) -> pd.Series:
    labels = [f'{lo}-{hi - 1}' if hi < 120 else f'>={lo}' for lo, hi in AGE_BANDS]
    edges = [b[0] for b in AGE_BANDS] + [np.inf]
    return pd.cut(age, edges, right=False, labels=labels)


def table1(cohort, suppress_below: int = 0) -> pd.DataFrame:
    """Characteristics by epsilon-4 allele count. Categorical rows: Pearson's chi-squared test
    of independence; follow-up: one-way ANOVA. Counts below `suppress_below` are masked."""
    P = cohort.persons.assign(age_group=_band_label(cohort.persons['last_age']))
    groups = [0, 1, 2]
    rows = [{'variable': 'N', 'level': '', **{f'e4_{g}': int((P['e4'] == g).sum()) for g in groups}}]
    for var in ['sex', 'race', 'age_group']:
        tab = pd.crosstab(P[var], P['e4']).reindex(columns=groups, fill_value=0)
        p = stats.chi2_contingency(tab)[1]
        for level, r in tab.iterrows():
            row = {'variable': var, 'level': level, 'test': "Pearson's chi-squared", 'p': p}
            for g in groups:
                n, tot = r[g], tab[g].sum()
                masked = 0 < n < suppress_below
                row[f'e4_{g}'] = f'<{suppress_below}' if masked else f'{n} ({100 * n / tot:.1f}%)'
            rows.append(row)
            p = np.nan                                  # report the test once per variable
    fu = [P.loc[P['e4'] == g, 'followup_years'] for g in groups]
    rows.append({'variable': 'EHR follow-up, years', 'level': 'mean (SD)',
                 **{f'e4_{g}': f'{x.mean():.2f} ({x.std():.2f})' for g, x in zip(groups, fu)},
                 'test': 'One-way ANOVA', 'p': stats.f_oneway(*fu).pvalue})
    return pd.DataFrame(rows)


def pc_variance(cohort) -> pd.DataFrame:
    """Share of the variance of the supplied PC scores carried by each PC. For PCA scores this
    equals each eigenvalue's share among the first N_PCS components (not of total genetic
    variance, which needs the full eigenvalue spectrum from the genotype PCA)."""
    v = cohort.persons[cohort.pc_cols].var()
    return pd.DataFrame({'pc': v.index, 'variance': v.values, 'share_among_pcs': v.values / v.sum()})
