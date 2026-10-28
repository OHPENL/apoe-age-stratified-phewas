"""Cohort construction and age-window tables, identical for every biobank.

Loaders hand over three standard tables:
  persons    : person_id, birth_date, sex ('M'/'F'), race (see RACE_LEVELS), geno ('E3_E4', ...), PC1..PCn
  encounters : person_id, date   (dated records that define EHR observation)
  events     : person_id, phecode, date
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (ADULT_AGE, CASE_SCOPE, GENO_TO_E4, MIN_ADULT_ENCOUNTER_DATES,
                     MIN_FOLLOWUP_DAYS, N_PCS, RACE_LEVELS)

RACE_DUMMIES = ['race_Black', 'race_Asian', 'race_Other', 'race_Unknown']   # White = reference


@dataclass
class Cohort:
    persons: pd.DataFrame          # index person_id
    enc: pd.DataFrame              # person_id, date, age (adult records of eligible persons)
    events: pd.DataFrame           # person_id, phecode, date, age
    pc_cols: list
    flow: dict = field(default_factory=dict)


def _years(delta: pd.Series) -> pd.Series:
    return delta.dt.days / 365.25


def build_cohort(persons: pd.DataFrame, encounters: pd.DataFrame, events: pd.DataFrame,
                 n_pcs: int = N_PCS) -> Cohort:
    pc_cols = [f'PC{i}' for i in range(1, n_pcs + 1)]
    p = persons.drop_duplicates('person_id').copy()
    flow = {'genotyped': len(p)}

    p['e4'] = p['geno'].map(GENO_TO_E4)
    flow['epsilon2_or_unknown_genotype_excluded'] = int(p['e4'].isna().sum())
    p = p[p['e4'].notna()]
    p = p[p['sex'].isin(['M', 'F']) & p['birth_date'].notna()].dropna(subset=pc_cols)
    flow['after_genotype_sex_pc'] = len(p)

    enc = encounters[['person_id', 'date']].drop_duplicates()
    enc = enc.merge(p[['person_id', 'birth_date']], on='person_id')
    enc['age'] = _years(enc['date'] - enc['birth_date'])
    enc = enc[enc['age'] >= ADULT_AGE]

    s = enc.groupby('person_id').agg(n_dates=('date', 'nunique'), first_date=('date', 'min'),
                                     last_date=('date', 'max'), first_age=('age', 'min'),
                                     last_age=('age', 'max'))
    span = (s['last_date'] - s['first_date']).dt.days
    s = s[(s['n_dates'] >= MIN_ADULT_ENCOUNTER_DATES) & (span >= MIN_FOLLOWUP_DAYS)].copy()
    s['followup_years'] = _years(s['last_date'] - s['first_date'])

    p = p.merge(s[['first_age', 'last_age', 'followup_years']], left_on='person_id', right_index=True)
    flow['eligible'] = len(p)
    p['race'] = p['race'].where(p['race'].isin(RACE_LEVELS), 'Other')
    p['male'] = (p['sex'] == 'M').astype(int)
    for col in RACE_DUMMIES:
        p[col] = (p['race'] == col.split('_', 1)[1]).astype(int)
    p = p.set_index('person_id')

    enc = enc[enc['person_id'].isin(p.index)][['person_id', 'date', 'age']]
    ev = events.merge(p[['birth_date']], left_on='person_id', right_index=True)
    ev['age'] = _years(ev['date'] - ev['birth_date'])
    ev = ev[ev['age'] >= ADULT_AGE][['person_id', 'phecode', 'date', 'age']]
    return Cohort(p, enc, ev, pc_cols, flow)


def age_windows(cohort: Cohort, bands, single_stratum: bool = False) -> dict:
    """band -> persons observed in the band, with age and follow-up measured inside the band.

    single_stratum=True keeps each person only in the band containing their last adult
    record, so no individual contributes to more than one stratum (Reviewer 3, comment 2).
    """
    out = {}
    for lo, hi in bands:
        e = cohort.enc[(cohort.enc['age'] >= lo) & (cohort.enc['age'] < hi)]
        g = e.groupby('person_id').agg(age_window=('age', 'max'), first=('date', 'min'),
                                       last=('date', 'max'))
        g['fu_window'] = _years(g['last'] - g['first'])
        w = cohort.persons.join(g[['age_window', 'fu_window']], how='inner')
        if single_stratum:
            w = w[(w['last_age'] >= lo) & (w['last_age'] < hi)]
        out[(lo, hi)] = w
    return out


def window_counts(cohort: Cohort, lo: float, hi: float, scope: str = CASE_SCOPE,
                  phecodes=None) -> pd.DataFrame:
    """Distinct diagnosis dates per person and PheCode within the window."""
    ev = cohort.events
    m = ev['age'] < hi
    if scope == 'window':
        m &= ev['age'] >= lo
    if phecodes is not None:
        m &= ev['phecode'].isin(phecodes)
    return (ev[m].groupby(['person_id', 'phecode'])['date'].nunique()
            .rename('n_dates').reset_index())


def lifetime_counts(cohort: Cohort, phecodes=None) -> pd.DataFrame:
    return window_counts(cohort, ADULT_AGE, np.inf, scope='cumulative', phecodes=phecodes)


def subgroup(win: pd.DataFrame, name: str, pc_cols: list):
    """Rows and covariates for one analysis group; the stratifying variable is not a covariate."""
    base = ['age_window', 'fu_window'] + pc_cols
    if name == 'Total':
        return win, base + ['male'] + RACE_DUMMIES
    if name in ('Male', 'Female'):
        return win[win['sex'] == name[0]], base + RACE_DUMMIES
    return win[win['race'] == name], base + ['male']
