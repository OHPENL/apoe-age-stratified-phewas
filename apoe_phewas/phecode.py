"""Map ICD-9-CM / ICD-10(-CM) diagnosis codes to PhecodeX."""
import pandas as pd

from .config import ICD10CM_START


def normalise_code(codes: pd.Series) -> pd.Series:
    """Upper-case, drop dots and UK Biobank 'X' fillers (e.g. 'C01X' -> 'C01')."""
    s = codes.astype(str).str.strip().str.upper().str.replace('.', '', regex=False)
    return s.str.replace(r'X+$', '', regex=True)


def load_phecode_map(source) -> pd.DataFrame:
    """PhecodeX flat map (columns ICD, vocabulary_id, phecode, phecode_string, category)."""
    m = source if isinstance(source, pd.DataFrame) else pd.read_csv(source, dtype=str, encoding='latin1')
    m = m.rename(columns=str.strip)
    m['vocab'] = m['vocabulary_id'].str.upper().map({'ICD9CM': 'ICD9', 'ICD10CM': 'ICD10'})
    m['code'] = normalise_code(m['ICD'])
    m = m.dropna(subset=['vocab', 'phecode'])
    return m[['vocab', 'code', 'phecode', 'phecode_string', 'category']].drop_duplicates()


def assign_vocab_by_date(dates: pd.Series) -> pd.Series:
    """For sources without a vocabulary column: ICD-9-CM before the US switch, ICD-10-CM after."""
    return pd.Series('ICD10', index=dates.index).where(dates >= pd.Timestamp(ICD10CM_START), 'ICD9')


def map_events(coded: pd.DataFrame, pmap: pd.DataFrame, truncate: bool = False) -> pd.DataFrame:
    """coded: person_id, code, vocab ('ICD9'/'ICD10'), date -> person_id, phecode, date.

    truncate=True (UK Biobank, WHO ICD-10): codes without an exact match in the ICD-10-CM
    map are retried at 4 and then 3 characters (e.g. WHO-only 'M511D' -> 'M511' -> 'M51')."""
    ev = coded.assign(code=normalise_code(coded['code']))
    ev = ev[~ev['code'].str.contains('-', regex=False)]          # ranges such as 'J450-J459'
    m = pmap[['vocab', 'code', 'phecode']]
    hit = ev.merge(m, on=['vocab', 'code'], how='inner')
    if truncate:
        rest = ev[~ev.set_index(['vocab', 'code']).index.isin(m.set_index(['vocab', 'code']).index)]
        for n in (4, 3):
            cand = rest.assign(code=rest['code'].str[:n])
            found = cand.merge(m, on=['vocab', 'code'], how='inner')
            hit = pd.concat([hit, found], ignore_index=True)
            rest = rest[~cand.set_index(['vocab', 'code']).index.isin(m.set_index(['vocab', 'code']).index)]
    return hit[['person_id', 'phecode', 'date']].drop_duplicates()


def phecode_labels(pmap: pd.DataFrame) -> pd.DataFrame:
    return pmap[['phecode', 'phecode_string', 'category']].drop_duplicates('phecode')
