"""Between-biobank consistency (Cochran's Q and I²) for PheCodes replicated in all three biobanks,
without pooling estimates (Reviewer 1, comment 8).

    python tools/heterogeneity.py  ->  out/combined/heterogeneity_replicated.csv
"""
import glob
from pathlib import Path

import pandas as pd
from scipy import stats

O = Path(__file__).resolve().parents[1] / 'out'


def main():
    m = pd.concat([pd.read_csv(f) for b in ('biovu', 'aou', 'ukb') for f in glob.glob(str(O / b / 'main_results*.csv'))])
    m = m[m['subgroup'] == 'Total']
    sig = m[(m['fdr'] < 0.05) & (m['or'] > 1)]
    nb = sig.groupby('phecode')['biobank'].nunique()
    codes = nb[nb == 3].index
    rows = []
    for (code, band), d in m[m['phecode'].isin(codes)].groupby(['phecode', 'age_band']):
        d = d.dropna(subset=['beta', 'se'])
        if d['biobank'].nunique() < 3:
            continue
        w = 1 / d['se'] ** 2
        b = d['beta']
        fixed = (w * b).sum() / w.sum()
        q = float((w * (b - fixed) ** 2).sum())
        dfree = len(d) - 1
        sig_here = sig[(sig['phecode'] == code) & (sig['age_band'] == band)]['biobank'].nunique() == 3
        rows.append({'phecode': code, 'phecode_string': d['phecode_string'].iloc[0], 'age_band': band,
                     'sig_all3': sig_here,
                     **{f'or_{r.biobank}': r['or'] for _, r in d.iterrows()},
                     'Q': q, 'df': dfree, 'p_Q': stats.chi2.sf(q, dfree),
                     'I2': max(0.0, (q - dfree) / q) * 100 if q > 0 else 0.0})
    out = pd.DataFrame(rows).sort_values(['phecode_string', 'age_band'])
    out.to_csv(O / 'combined' / 'heterogeneity_replicated.csv', index=False)
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
