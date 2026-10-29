"""Revision 2 extras for UK Biobank: HWE and record dates, 10-PC refit, primary care coverage.

    python tools/ukb_rev2.py /path/to/ukb
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apoe_phewas.cohort import build_cohort                      # noqa: E402
from apoe_phewas.phecode import load_phecode_map, map_events     # noqa: E402
from loaders.ukb import PLACEHOLDER_DATES, load                  # noqa: E402
from run_biobank import run                                      # noqa: E402

UKB = Path(sys.argv[1] if len(sys.argv) > 1 else '/path/to/ukb')
MAP = UKB / 'phecodeX_ICD_CM_map_flat.csv'

if __name__ == '__main__':
    persons, enc, coded = load(UKB)
    cohort = build_cohort(persons, enc, map_events(coded, load_phecode_map(MAP), truncate=True))
    P = cohort.persons
    ids = set(P.index) if P.index.name == 'person_id' else set(P['person_id'])
    src = pd.read_csv(UKB / 'condition_uk_officialmap.csv', usecols=['person_id', 'visit_start_date', 'source'])
    src['date'] = pd.to_datetime(src['visit_start_date'], errors='coerce')
    src = src[src['date'].notna() & ~src['date'].isin(PLACEHOLDER_DATES) & src['person_id'].isin(ids)]
    gp_ids = set(src.loc[src['source'] == 'gp', 'person_id'])
    rows = [{'n_analytic': len(ids), 'n_with_primary_care': len(gp_ids),
             'pct_with_primary_care': round(100 * len(gp_ids) / len(ids), 1)}]
    for s, d in src.groupby('source'):
        rows.append({'source': s, 'first': d['date'].min().date(), 'last': d['date'].max().date(),
                     'p99_last': d['date'].quantile(0.99).date()})
    pd.DataFrame(rows).to_csv('out/ukb/ukb_sources_rev2.csv', index=False)
    print(pd.DataFrame(rows))
    run('ukb', persons, enc, coded, MAP, 'out/ukb', analyses=('hwe', 'pc10'), part='rev2')
    print('done')
