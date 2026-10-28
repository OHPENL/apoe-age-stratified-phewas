"""Combine the three biobanks' summary statistics into paper / response-letter tables.

    python summarise_all.py --biovu out/biovu --aou out/aou --ukb out/ukb --out out/combined
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from apoe_phewas.config import FDR_ALPHA

KEYS = ['subgroup', 'age_band', 'phecode']


def _read(dirs, name):
    # also picks up split-run parts such as main_results_A.csv / main_results_B.csv
    parts = [pd.read_csv(f) for d in dirs for f in sorted(d.glob(f'{name}*.csv'))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def replication_summary(main):
    """53/22/43/12/15/9-style counts per subgroup (positive associations, FDR < 0.05)."""
    sig = main[(main['fdr'] < FDR_ALPHA) & (main['or'] > 1)]
    rows = []
    for g, s in sig.groupby('subgroup'):
        n_bb = s.groupby('phecode')['biobank'].nunique()
        rows.append({'subgroup': g, 'unique_phecodes': len(n_bb),
                     **{f'n_{b}': s.loc[s['biobank'] == b, 'phecode'].nunique()
                        for b in sorted(main['biobank'].unique())},
                     'replicated_ge2': int((n_bb >= 2).sum()), 'replicated_all3': int((n_bb >= 3).sum())})
    return pd.DataFrame(rows)


def compare(main, other, label):
    """How significant main-analysis associations behave in a sensitivity analysis."""
    sig = main[main['fdr'] < FDR_ALPHA]
    m = sig.merge(other, on=['biobank'] + KEYS, suffixes=('', f'_{label}'))
    lm, lo = np.log(m['or']), np.log(m[f'or_{label}'])
    pcol = f'fdr_{label}' if f'fdr_{label}' in m.columns else f'p_{label}'
    n = len(m)
    summary = {'analysis': label, 'n_compared': n, 'n_significant_main': len(sig),
               'corr_log_or': np.corrcoef(lm, lo)[0, 1] if n > 1 else np.nan,
               'median_abs_diff_log_or': float(np.median(np.abs(lm - lo))) if n else np.nan,
               'same_direction_pct': 100 * float(np.mean(np.sign(lm) == np.sign(lo))) if n else np.nan,
               f'still_{pcol}_lt_0.05_pct': 100 * float(np.mean(m[pcol] < FDR_ALPHA)) if n else np.nan}
    return m, summary


def _interaction_fdr(inter):
    """BH across PheCodes within each biobank, separately for each interaction test."""
    for col in ('p_e4_x_age', 'p_e4_x_sex', 'p_e4_x_race'):
        out = col.replace('p_', 'fdr_')
        inter[out] = np.nan
        for _, idx in inter.groupby('biobank').groups.items():
            ids = [i for i in idx if pd.notna(inter.at[i, col])]
            if ids:
                inter.loc[ids, out] = multipletests(inter.loc[ids, col], method='fdr_bh')[1]
    return inter


def main():
    ap = argparse.ArgumentParser()
    for b in ('biovu', 'aou', 'ukb'):
        ap.add_argument(f'--{b}', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    dirs = [d for d in (a.biovu, a.aou, a.ukb) if d is not None]
    a.out.mkdir(parents=True, exist_ok=True)

    main_res = _read(dirs, 'main_results')
    # One BH over every main-analysis test per biobank (recomputed so split runs are pooled).
    main_res['fdr_global'] = np.nan
    for _, idx in main_res.groupby('biobank').groups.items():
        ids = [i for i in idx if pd.notna(main_res.at[i, 'p'])]
        main_res.loc[ids, 'fdr_global'] = multipletests(main_res.loc[ids, 'p'], method='fdr_bh')[1]
    sig = main_res['fdr'] < FDR_ALPHA
    replication_summary(main_res).to_csv(a.out / 'replication_summary.csv', index=False)
    main_res[sig & (main_res['or'] > 1)].to_csv(a.out / 'significant_positive.csv', index=False)
    # Reviewer 3, comment 7: protective associations.
    main_res[sig & (main_res['or'] < 1)].to_csv(a.out / 'protective_associations.csv', index=False)
    # Reviewer 5, comment 4: every diabetes PheCode tested (PhecodeX EM_202.*).
    main_res[main_res['phecode'].astype(str).str.startswith('EM_202')].to_csv(
        a.out / 'diabetes_results.csv', index=False)
    # Reviewer 3, comment 3: within-scan versus one global BH correction.
    (main_res.assign(sig_within=sig, sig_global=main_res['fdr_global'] < FDR_ALPHA)
     .groupby(['biobank', 'subgroup'])[['sig_within', 'sig_global']].sum().reset_index()
     .to_csv(a.out / 'fdr_family_comparison.csv', index=False))
    # Stage-2 targets: PheCodes significant in the Total analysis of any biobank.
    tgt = main_res[(main_res['subgroup'] == 'Total') & sig]
    pd.DataFrame({'phecode': sorted(tgt['phecode'].unique())}).to_csv(a.out / 'targets.csv', index=False)

    summaries = []
    for name, label in [('firth_results', 'firth'), ('single_stratum_results', 'single')]:
        other = _read(dirs, name)
        if other.empty:
            continue
        cols = ['biobank'] + KEYS + [c for c in ('or', 'or_lo', 'or_hi', 'p', 'fdr') if c in other.columns]
        merged, s = compare(main_res, other[cols], label)
        merged.to_csv(a.out / f'{label}_vs_main.csv', index=False)
        summaries.append(s)
    if summaries:
        pd.DataFrame(summaries).to_csv(a.out / 'sensitivity_summary.csv', index=False)

    b10 = _read(dirs, 'bands10_results')
    if not b10.empty:
        b10[b10['fdr'] < FDR_ALPHA].to_csv(a.out / 'bands10_significant.csv', index=False)
    # Interaction / continuous-age models: keep only the current target PheCodes (a biobank may
    # also hold results for targets from an earlier run, plus '_extra' runs for new ones).
    targets = set(tgt['phecode'])
    inter = _read(dirs, 'interaction_results')
    if not inter.empty:
        inter = inter[inter['phecode'].isin(targets)].drop_duplicates(['biobank', 'phecode'], keep='last')
        _interaction_fdr(inter.reset_index(drop=True)).to_csv(a.out / 'interaction_results_all.csv', index=False)
    cont = _read(dirs, 'continuous_age_results')
    if not cont.empty:
        cont = cont[cont['phecode'].isin(targets)]
        cont.to_csv(a.out / 'continuous_age_results_all.csv', index=False)
    lin = _read(dirs, 'linear_age_results')
    if not lin.empty:
        lin = lin[lin['phecode'].isin(targets)].reset_index(drop=True)
        lin['fdr_e4_x_age_linear'] = np.nan
        for _, idx in lin.groupby('biobank').groups.items():
            ids = [i for i in idx if pd.notna(lin.at[i, 'p_e4_x_age_linear'])]
            if ids:
                lin.loc[ids, 'fdr_e4_x_age_linear'] = multipletests(
                    lin.loc[ids, 'p_e4_x_age_linear'], method='fdr_bh')[1]
        lin.to_csv(a.out / 'linear_age_results_all.csv', index=False)
    for name in ('cohort_flow', 'table1', 'pc_variance'):
        t = _read(dirs, name)
        if not t.empty:
            t.to_csv(a.out / f'{name}_all.csv', index=False)
    print('wrote', sorted(p.name for p in a.out.glob('*.csv')))


if __name__ == '__main__':
    main()
