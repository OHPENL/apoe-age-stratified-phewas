"""VUMC BioVU loader for the Databricks workspace (same tables as APOE_ordinal2.ipynb)."""
import pandas as pd

from apoe_phewas.config import N_PCS
from apoe_phewas.phecode import assign_vocab_by_date

TABLES = {
    'person': 'victr_sd.sd_omop_prod.person',
    'concept': 'victr_sd.sd_omop_prod.concept',
    'condition': 'victr_sd.sd_omop_prod.condition_occurrence',
    'geno': 'CATALOG.SCHEMA.adrd_250_k_geno',
    'pcs': 'CATALOG.SCHEMA.agd_250_k_pc',
    'ids': 'CATALOG.SCHEMA.agd_250k_ids_person_id',
    'phecode_map': 'CATALOG.SCHEMA.phecode_x_icd_cm_map_flat',
}
RACE = {'W': 'White', 'B': 'Black', 'A': 'Asian', 'U': 'Unknown'}
VOCAB = {'ICD9CM': 'ICD9', 'ICD10CM': 'ICD10'}


def load(spark, n_pcs: int = N_PCS, min_date: str = '2000-01-01', tables: dict = TABLES,
         years_per_chunk: int = 3):
    """Returns persons, encounters, coded events and the PhecodeX map (pandas)."""
    t = tables
    person = spark.sql(f"""SELECT person_source_value AS psv, birth_datetime, gender_source_value,
                           race_source_value FROM {t['person']}""").toPandas()
    geno = spark.sql(f"SELECT PERSON_SOURCE_VALUE AS psv, GENO FROM {t['geno']}").toPandas()
    pc_sql = ', '.join(f'PC{i}_SUM' for i in range(1, n_pcs + 1))
    pcs = spark.sql(f"SELECT GRID AS psv, {pc_sql} FROM {t['pcs']}").toPandas()
    p = person.merge(geno, on='psv').merge(pcs, on='psv').drop_duplicates('psv')
    persons = pd.DataFrame({
        'person_id': p['psv'],
        'birth_date': pd.to_datetime(p['birth_datetime'], errors='coerce'),
        'sex': p['gender_source_value'],
        'race': p['race_source_value'].map(RACE).fillna('Other'),
        'geno': p['GENO'],
        **{f'PC{i}': p[f'PC{i}_SUM'].astype(float) for i in range(1, n_pcs + 1)}})

    # The source concept gives the code's vocabulary; codes without an ICD source concept
    # fall back to the date of the US ICD-9-CM -> ICD-10-CM switch.
    # Pulled one block of years at a time so no single collect exceeds spark.driver.maxResultSize.
    start, stop = int(min_date[:4]), pd.Timestamp.today().year + 1
    parts = []
    for y0 in range(start, stop, years_per_chunk):
        y1 = min(y0 + years_per_chunk, stop)
        parts.append(spark.sql(f"""
            SELECT ids.PERSON_SOURCE_VALUE AS person_id, v.condition_source_value AS code,
                   v.condition_start_date AS date,
                   CASE c.vocabulary_id WHEN 'ICD9CM' THEN 'ICD9' WHEN 'ICD10CM' THEN 'ICD10' END AS vocab
            FROM {t['condition']} v
            JOIN {t['ids']} ids ON ids.person_id = v.person_id
            LEFT JOIN {t['concept']} c ON c.concept_id = v.condition_source_concept_id
            WHERE v.condition_start_date > '{min_date}'
              AND v.condition_start_date >= '{y0}-01-01' AND v.condition_start_date < '{y1}-01-01'
            """).toPandas())
        print(f'  conditions {y0}-{y1 - 1}: {len(parts[-1]):,} rows')
    cond = pd.concat(parts, ignore_index=True)
    cond['date'] = pd.to_datetime(cond['date'], errors='coerce')
    cond = cond.dropna(subset=['date', 'code'])
    missing = cond['vocab'].isna()
    print(f'vocabulary from source concept: {1 - missing.mean():.1%}; date rule for the rest')
    cond.loc[missing, 'vocab'] = assign_vocab_by_date(cond.loc[missing, 'date'])
    coded = cond[['person_id', 'code', 'vocab', 'date']]
    encounters = cond[['person_id', 'date']].drop_duplicates()
    # tables['phecode_map'] = None: the caller supplies the PhecodeX CSV instead.
    pmap = spark.sql(f"SELECT * FROM {t['phecode_map']}").toPandas() if t.get('phecode_map') else None
    return persons, encounters, coded, pmap
