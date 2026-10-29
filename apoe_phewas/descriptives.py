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


GENOTYPES = ('E2_E2', 'E2_E3', 'E2_E4', 'E3_E3', 'E3_E4', 'E4_E4')


def hwe_by_race(persons: pd.DataFrame, suppress_below: int = 0) -> pd.DataFrame:
    """APOE genotype counts and the three-allele Hardy-Weinberg chi-squared test (df = 3) by
    self-reported race, in the same genotyped sample that Figure 1 counts (before exclusions)."""
    p = persons.drop_duplicates('person_id')
    p = p[p['geno'].isin(GENOTYPES)]
    rows = []
    for race, d in list(p.groupby('race')) + [('All', p)]:
        obs = {g: int((d['geno'] == g).sum()) for g in GENOTYPES}
        n = sum(obs.values())
        alleles = {a: sum(c * g.count(a) for g, c in obs.items()) for a in ('E2', 'E3', 'E4')}
        f = {a: alleles[a] / (2 * n) for a in alleles}
        exp = {g: n * (f[g[:2]] ** 2 if g[:2] == g[3:] else 2 * f[g[:2]] * f[g[3:]]) for g in GENOTYPES}
        chi = sum((obs[g] - exp[g]) ** 2 / exp[g] for g in GENOTYPES if exp[g] > 0)
        k = sum(alleles[a] > 0 for a in alleles)
        dfree = k * (k - 1) // 2
        row = {'race': race, 'N': n, **{f'n_{g}': obs[g] for g in GENOTYPES},
               **{f'freq_{a}': round(f[a], 4) for a in f}, 'hwe_chi2': round(chi, 2), 'hwe_df': dfree,
               'hwe_p': stats.chi2.sf(chi, dfree) if dfree else np.nan}
        if suppress_below:                               # All of Us dissemination policy
            for key in ['N'] + [f'n_{g}' for g in GENOTYPES]:
                if 1 <= row[key] < suppress_below:
                    row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def record_dates(cohort) -> pd.DataFrame:
    """First and last adult EHR record dates in the analytic sample (observation period)."""
    d = cohort.enc['date']
    return pd.DataFrame([{'first_record': d.min().date(), 'last_record': d.max().date(),
                          'p99_last_record': d.quantile(0.99).date()}])
