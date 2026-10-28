# All of Us Researcher Workbench — APOE e4 revision analyses (Python Jupyter notebook,
# Controlled Tier). Paste each "# %%" block into its own cell and run in order.
# Only summary statistics are written; counts 1-19 are masked automatically.

# %%
# 1) Unpack the pipeline uploaded to the notebook's home directory.
# Run this cell first after every kernel restart: thread settings must precede numpy.
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, zipfile
zipfile.ZipFile('revision_pipeline_upload.zip').extractall('.')
sys.path.insert(0, 'revision_pipeline')
print(sorted(os.listdir('revision_pipeline')))

# %%
# 2) Ancestry PCs from this workspace's CDR (path follows the workspace CDR version)
CDR_STORAGE = os.environ['CDR_STORAGE_PATH']
ANCESTRY_GS = f'{CDR_STORAGE}/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv'
print(ANCESTRY_GS)
os.system(f'gsutil -u {os.environ["GOOGLE_PROJECT"]} cp {ANCESTRY_GS} ancestry_preds.tsv')
print(os.path.exists('ancestry_preds.tsv'))

# %%
# 3) APOE genotypes: a CSV with person_id, GENO ('E3_E3', 'E3_E4', 'E4_E4', 'E2_E3', ...).
# If you still have the earlier analysis_base.csv in this workspace, this builds it:
import pandas as pd
if not os.path.exists('apoe_genotypes.csv'):
    (pd.read_csv('analysis_base.csv', usecols=['person_id', 'GENO'])
       .drop_duplicates('person_id').to_csv('apoe_genotypes.csv', index=False))
print(pd.read_csv('apoe_genotypes.csv')['GENO'].value_counts())

# %%
# 4) Load data from the CDR (BigQuery)
from loaders.aou import load
from run_biobank import run
PHECODE_MAP = 'revision_pipeline/phecodeX_ICD_CM_map_flat.csv'
OUT_DIR = 'apoe_revision_aou'
persons, encounters, coded = load('apoe_genotypes.csv', 'ancestry_preds.tsv')
print(len(persons), len(encounters), len(coded))

# %%
# 5) Stage 1
run('aou', persons, encounters, coded, PHECODE_MAP, OUT_DIR,
    analyses=('table1', 'pcs', 'main', 'bands10', 'single'))

# %%
# 6) Stage 2 — only after targets.csv (from summarise_all.py) has been uploaded into OUT_DIR
run('aou', persons, encounters, coded, PHECODE_MAP, OUT_DIR,
    analyses=('firth', 'interaction', 'continuous'), targets=os.path.join(OUT_DIR, 'targets.csv'))

# %%
# 7) Bundle the summary CSVs for download (check them against the dissemination policy first)
import shutil
shutil.make_archive('apoe_revision_aou_results', 'zip', OUT_DIR)
