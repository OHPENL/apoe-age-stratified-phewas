"""UK Biobank loader for the locally extracted files used in the original analysis."""
from pathlib import Path

import pandas as pd

from apoe_phewas.config import N_PCS

RACE = {'White': 'White', 'Black': 'Black', 'Asian': 'Asian', 'Unknown': 'Unknown'}
# UK Biobank special dates meaning "unknown"/"before birth"/"future".
PLACEHOLDER_DATES = pd.to_datetime(['1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03',
                                    '1909-09-09', '2037-07-07'])


def load(data_dir, n_pcs: int = N_PCS, conditions_file: str = 'condition_uk_officialmap.csv'):
    d = Path(data_dir)
    person = pd.read_csv(d / 'person_uk.csv')
    geno = pd.read_csv(d / 'adrd_uk_geno.csv', usecols=['person_id', 'GENO'])
    p = person.merge(geno, on='person_id', how='inner')
    persons = pd.DataFrame({
        'person_id': p['person_id'],
        # Only year of birth is released; mid-year is used as the birth date.
        'birth_date': pd.to_datetime(p['birth_datetime'].astype('Int64').astype(str) + '-07-01',
                                     errors='coerce'),
        'sex': p['gender_source_value'].map({1: 'M', 0: 'F'}),
        'race': p['race_source_value'].map(RACE).fillna('Other'),
        'geno': p['GENO'],
        **{f'PC{i}': p[f'PC{i}_SUM'] for i in range(1, n_pcs + 1)}})

    # condition_uk_officialmap.csv: built by tools/build_ukb_conditions.py (one primary
    # Read -> ICD-10 mapping per code); the older condition_uk.csv kept every candidate.
    cond = pd.read_csv(d / conditions_file, dtype={'condition_source_value': str},
                       usecols=['person_id', 'condition_source_value', 'visit_start_date'])
    cond = cond.rename(columns={'condition_source_value': 'code', 'visit_start_date': 'date'})
    cond['date'] = pd.to_datetime(cond['date'], errors='coerce')
    cond = cond[cond['date'].notna() & ~cond['date'].isin(PLACEHOLDER_DATES)]
    cond = cond[cond['person_id'].isin(persons['person_id'])]
    # UK Biobank records use WHO ICD-10, matched to the ICD-10-CM rows of the PhecodeX map.
    coded = cond.assign(vocab='ICD10')[['person_id', 'code', 'vocab', 'date']]
    encounters = cond[['person_id', 'date']].drop_duplicates()
    return persons, encounters, coded
