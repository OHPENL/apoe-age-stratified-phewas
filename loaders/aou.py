"""All of Us loader for the Researcher Workbench (Controlled Tier CDR via BigQuery).

APOE genotypes come from your existing rs429358/rs7412 extraction (CSV: person_id, GENO).
Ancestry PCs come from the CDR's ancestry_preds.tsv (16 PCs in `pca_features`).
"""
import ast
import os

import pandas as pd

from apoe_phewas.config import N_PCS

SEX = {'Male': 'M', 'Female': 'F'}
RACE = {'White': 'White', 'Black or African American': 'Black', 'Asian': 'Asian',
        'None Indicated': 'Unknown', 'I prefer not to answer': 'Unknown', 'PMI: Skip': 'Unknown',
        'No matching concept': 'Unknown'}
VOCAB = {'ICD9CM': 'ICD9', 'ICD10CM': 'ICD10'}


def _query(sql: str) -> pd.DataFrame:
    from google.cloud import bigquery
    return bigquery.Client().query(sql).to_dataframe()


def load(genotype_csv: str, ancestry_tsv: str, n_pcs: int = N_PCS, cdr: str = None):
    cdr = cdr or os.environ['WORKSPACE_CDR']
    person = _query(f"""
        SELECT p.person_id, p.birth_datetime, sx.concept_name AS sex, rc.concept_name AS race
        FROM `{cdr}.person` p
        LEFT JOIN `{cdr}.concept` sx ON sx.concept_id = p.sex_at_birth_concept_id
        LEFT JOIN `{cdr}.concept` rc ON rc.concept_id = p.race_concept_id""")
    geno = pd.read_csv(genotype_csv, usecols=['person_id', 'GENO'])
    anc = pd.read_csv(ancestry_tsv, sep='\t', usecols=['research_id', 'pca_features'])
    pcs = pd.DataFrame(anc['pca_features'].map(ast.literal_eval).tolist()).iloc[:, :n_pcs]
    pcs.columns = [f'PC{i}' for i in range(1, n_pcs + 1)]
    pcs['person_id'] = anc['research_id'].to_numpy()

    p = person.merge(geno, on='person_id').merge(pcs, on='person_id')
    persons = pd.DataFrame({
        'person_id': p['person_id'],
        'birth_date': pd.to_datetime(p['birth_datetime'], errors='coerce', utc=True).dt.tz_localize(None),
        'sex': p['sex'].map(SEX),
        'race': p['race'].map(RACE).fillna('Other'),
        'geno': p['GENO'],
        **{f'PC{i}': p[f'PC{i}'] for i in range(1, n_pcs + 1)}})

    # ICD source codes from condition_occurrence and observation (the standard PheWAS extraction).
    coded = _query(f"""
        SELECT co.person_id, c.concept_code AS code, c.vocabulary_id, co.condition_start_date AS date
        FROM `{cdr}.condition_occurrence` co
        JOIN `{cdr}.concept` c ON c.concept_id = co.condition_source_concept_id
        WHERE c.vocabulary_id IN ('ICD9CM', 'ICD10CM')
        UNION DISTINCT
        SELECT o.person_id, c.concept_code, c.vocabulary_id, o.observation_date
        FROM `{cdr}.observation` o
        JOIN `{cdr}.concept` c ON c.concept_id = o.observation_source_concept_id
        WHERE c.vocabulary_id IN ('ICD9CM', 'ICD10CM')""")
    coded = coded[coded['person_id'].isin(persons['person_id'])]
    coded['date'] = pd.to_datetime(coded['date'], errors='coerce')
    coded['vocab'] = coded['vocabulary_id'].map(VOCAB)
    coded = coded.dropna(subset=['date'])[['person_id', 'code', 'vocab', 'date']]
    encounters = coded[['person_id', 'date']].drop_duplicates()
    return persons, encounters, coded
