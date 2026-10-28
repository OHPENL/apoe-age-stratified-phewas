"""End-to-end check on synthetic data: every analysis runs and a planted effect is recovered.

    python tests/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_biobank import run  # noqa: E402

RNG = np.random.default_rng(1)
N = 6000
# PheCode -> (ICD-10-CM code, log-OR per epsilon-4 allele, baseline log-odds, effect only after age 60)
PHENOS = {'EM_239': ('E78.5', np.log(1.8), -2.0, False),
          'NS_328.11': ('G30.9', np.log(2.5), -2.5, True),
          'GI_530': ('K21.9', 0.0, -2.0, False)}


def synthetic():
    birth = pd.to_datetime(RNG.integers(1930, 2000, N).astype(str) + '-06-15')
    geno = RNG.choice(['E3_E3', 'E3_E4', 'E4_E4', 'E2_E3'], N, p=[0.62, 0.25, 0.03, 0.10])
    persons = pd.DataFrame({'person_id': np.arange(N), 'birth_date': birth,
                            'sex': RNG.choice(['M', 'F'], N),
                            'race': RNG.choice(['White', 'Black', 'Asian', 'Other', 'Unknown'], N,
                                               p=[0.7, 0.15, 0.05, 0.05, 0.05]),
                            'geno': geno,
                            **{f'PC{i}': RNG.normal(size=N) for i in range(1, 17)}})
    e4 = pd.Series(geno).map({'E3_E3': 0, 'E3_E4': 1, 'E4_E4': 2}).fillna(0).to_numpy()

    enc_rows, ev_rows = [], []
    for i in range(N):
        start = max(birth[i] + pd.DateOffset(years=18), pd.Timestamp('2000-01-01'))
        span = (pd.Timestamp('2023-12-31') - start).days
        if span < 400:
            continue
        dates = start + pd.to_timedelta(np.sort(RNG.integers(0, span, RNG.integers(3, 25))), 'D')
        enc_rows += [(i, d) for d in dates]
        for code, (icd, logor, base, late) in PHENOS.items():
            for d in dates:
                age = (d - birth[i]).days / 365.25
                eff = logor * e4[i] if (not late or age >= 60) else 0.0
                if RNG.random() < 1 / (1 + np.exp(-(base - 1.5 + eff))):
                    ev_rows.append((i, icd, d))
    encounters = pd.DataFrame(enc_rows, columns=['person_id', 'date'])
    coded = pd.DataFrame(ev_rows, columns=['person_id', 'code', 'date']).assign(vocab='ICD10')
    pmap = pd.DataFrame({'ICD': [v[0] for v in PHENOS.values()], 'vocabulary_id': 'ICD10CM',
                         'phecode': list(PHENOS), 'phecode_string': list(PHENOS), 'category': 'test'})
    return persons, encounters, coded, pmap


def main():
    persons, encounters, coded, pmap = synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'ukb'
        run('ukb', persons, encounters, coded, pmap, out,
            subgroups=('Total', 'Male', 'Female', 'White', 'Black'), targets=list(PHENOS))
        files = sorted(p.name for p in out.glob('*.csv'))
        print(files)
        expected = {'cohort_flow.csv', 'table1.csv', 'pc_variance.csv', 'main_results.csv',
                    'bands10_results.csv', 'single_stratum_results.csv', 'firth_results.csv',
                    'interaction_results.csv', 'continuous_age_results.csv',
                    'linear_age_results.csv'}
        assert expected <= set(files), expected - set(files)

        main_res = pd.read_csv(out / 'main_results.csv')
        tot = main_res[main_res['subgroup'] == 'Total'].set_index(['phecode', 'age_band'])
        print(tot[['n_cases', 'or', 'p', 'fdr', 'method']].round(4))
        assert (tot.loc['EM_239', 'or'] > 1.4).all() and (tot.loc['EM_239', 'fdr'] < 0.05).all()
        assert tot.loc[('NS_328.11', '18-44'), 'or'] < 1.5          # no effect before 60
        assert tot.loc[('NS_328.11', '75+'), 'fdr'] < 0.05

        firth = pd.read_csv(out / 'firth_results.csv')
        merged = firth.merge(main_res, on=['subgroup', 'age_band', 'phecode'], suffixes=('_f', '_m'))
        lf, lm = np.log(merged['or_f']), np.log(merged['or_m'])
        assert (np.abs(lf) <= np.abs(lm) + 1e-9).all()                 # Firth shrinks towards 0
        big = merged['n_cases_m'] >= 80
        assert np.allclose(lf[big], lm[big], atol=0.1)                 # and agrees when cases are many

        inter = pd.read_csv(out / 'interaction_results.csv').set_index('phecode')
        print(inter[['status', 'p_e4_x_age', 'p_e4_x_sex', 'p_e4_x_race']])
        assert inter.loc['NS_328.11', 'p_e4_x_age'] < 0.05             # planted age interaction
        cont = pd.read_csv(out / 'continuous_age_results.csv')
        print(cont[cont['phecode'] == 'NS_328.11'][['age', 'or', 'p_e4_x_age_spline']].round(3))
        lin = pd.read_csv(out / 'linear_age_results.csv').set_index('phecode')
        print(lin[['status', 'ratio_per_decade', 'p_e4_x_age_linear']])
        assert lin.loc['NS_328.11', 'ratio_per_decade'] > 1
        assert lin.loc['NS_328.11', 'p_e4_x_age_linear'] < 0.05
        print(pd.read_csv(out / 'table1.csv').to_string())
        print(pd.read_csv(out / 'cohort_flow.csv').to_string())
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
