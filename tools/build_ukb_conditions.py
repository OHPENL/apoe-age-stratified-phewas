"""Rebuild UK Biobank diagnosis events with one primary Read -> ICD-10 mapping per code.

The earlier condition_uk.csv kept every candidate ICD-10 code listed for a Read code
(e.g. C10.. -> E10-E14 + O24.9 + P70.x), creating spurious pregnancy/neonatal diagnoses.
Rules (UK Biobank all_lkps_maps_v4.xlsx):
  * read_v2_icd10: one official mapping per Read v2 code; multi-code entries (comma,
    dagger/asterisk, '+') are split; ICD-10 ranges (icd10_code_def 2) are not used.
  * read_ctv3_icd10 (codes not resolved above): mapping_status E (exact), G (general) or
    D (default) in block 0 only; R (requires checking), A (alternative) and alternative
    blocks are dropped.
  * Hospital (HES) records are already ICD-10 and are used as recorded.

    python tools/build_ukb_conditions.py --ukb-dir /path/to/ukb
"""
import argparse
import re
from pathlib import Path

import pandas as pd

MARKER = re.compile(r'^([A-Z]\d{2}[0-9X]?)[DA]$')   # dagger (D) / asterisk (A) suffix


def _clean(code: str):
    c = code.strip().upper().replace('.', '')
    m = MARKER.match(c)
    return m.group(1) if m else c


def read_mapping(lkps: Path) -> dict:
    v2 = pd.read_excel(lkps, sheet_name='read_v2_icd10', dtype=str).dropna(subset=['icd10_code'])
    c3 = pd.read_excel(lkps, sheet_name='read_ctv3_icd10', dtype=str).dropna(subset=['icd10_code'])

    c3 = c3[c3['mapping_status'].isin(['E', 'G', 'D']) & (c3['block_num'] == '0')]
    mapping = {k: sorted({_clean(x) for x in g}) for k, g in c3.groupby('read_code')['icd10_code']}

    v2 = v2[v2['icd10_code_def'] != '2']                               # ranges
    for code, icd in zip(v2['read_code'], v2['icd10_code']):
        parts = [p for p in re.split(r'[ ,+]+', icd) if p and '-' not in p]
        if parts:
            mapping[code] = sorted({_clean(p) for p in parts})        # v2 takes priority
    return mapping


def build(ukb_dir: Path, out: Path, chunksize: int = 5_000_000):
    mapping = read_mapping(ukb_dir / 'all_lkps_maps_v4.xlsx')
    print(f'Read codes with a primary mapping: {len(mapping):,}')
    first = True
    stats = {'gp_rows': 0, 'gp_unmapped': 0, 'gp_events': 0, 'hes_events': 0}
    for f in sorted(ukb_dir.glob('condition_out_patients_[0-9].csv')):
        for ch in pd.read_csv(f, dtype=str, chunksize=chunksize):
            ch = ch.dropna(subset=['diagnosis', 'diagnosis_date'])
            stats['gp_rows'] += len(ch)
            ch['icd'] = ch['diagnosis'].map(mapping)
            stats['gp_unmapped'] += int(ch['icd'].isna().sum())
            ev = ch.dropna(subset=['icd']).explode('icd')
            ev = ev.rename(columns={'eid': 'person_id', 'icd': 'condition_source_value',
                                    'diagnosis_date': 'visit_start_date'})
            ev = ev[['person_id', 'condition_source_value', 'visit_start_date']].assign(source='gp')
            ev.drop_duplicates().to_csv(out, mode='w' if first else 'a', header=first, index=False)
            first = False
            stats['gp_events'] += len(ev)
        print(f'  {f.name} done')
    hes = pd.read_csv(ukb_dir / 'condition_in_patients.csv', dtype=str).dropna(
        subset=['diagnosis', 'diagnosis_date'])
    hes = pd.DataFrame({'person_id': hes['eid'], 'condition_source_value': hes['diagnosis'].map(_clean),
                        'visit_start_date': hes['diagnosis_date'], 'source': 'hes'}).drop_duplicates()
    hes.to_csv(out, mode='a', header=False, index=False)
    stats['hes_events'] = len(hes)
    print(stats)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ukb-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    build(a.ukb_dir, a.out or a.ukb_dir / 'condition_uk_officialmap.csv')
