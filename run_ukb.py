"""UK Biobank, run locally.

    python run_ukb.py --data-dir /path/to/ukb --out out/ukb --stage 1
    python run_ukb.py --data-dir /path/to/ukb --out out/ukb --stage 2 --targets out/combined/targets.csv
"""
import argparse
from pathlib import Path

from loaders.ukb import load
from run_biobank import run

STAGES = {1: ('table1', 'pcs', 'main', 'bands10', 'single'),
          2: ('firth', 'interaction', 'continuous')}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage', type=int, choices=[1, 2], required=True)
    ap.add_argument('--targets', type=Path)
    ap.add_argument('--phecode-map', type=Path)
    a = ap.parse_args()
    persons, encounters, coded = load(a.data_dir)
    run('ukb', persons, encounters, coded, a.phecode_map or a.data_dir / 'phecodeX_ICD_CM_map_flat.csv',
        a.out, analyses=STAGES[a.stage], targets=a.targets)
