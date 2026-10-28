# Databricks notebook source
# VUMC BioVU — APOE e4 revision analyses. Run the cells in order.
# Before running: the unzipped revision_pipeline files are in
# Workspace > Users > eugene.jeong.1@vumc.org > revision_pipeline

# COMMAND ----------
# 1) Paths. Run this cell first after every Python restart: the thread settings must be in
# place before numpy is imported.
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
UPLOAD_DIR = '/Workspace/Users/eugene.jeong.1@vumc.org/revision_pipeline'
OUT_DIR = '/Workspace/Users/eugene.jeong.1@vumc.org/apoe_revision_biovu'   # summary CSVs are written here
# Split across two environments: 'A' in WORKSPACE_A, 'B' in WORKSPACE_B ('' = run everything here)
PART = 'A'
# Catalog holding the genotype / PC / ID / PhecodeX tables in THIS environment
WORKSPACE_CATALOG = 'WORKSPACE_A'      # second environment: 'WORKSPACE_B'
# Any other table name that differs here (optional). The PhecodeX map is read from the CSV in
# the uploaded folder so all three biobanks use the identical file.
TABLE_OVERRIDES = {'phecode_map': None}

# COMMAND ----------
# 2) Find the code (works whether or not unzipping added an extra revision_pipeline level)
import os, sys
CODE_DIR = next((root for root, _, files in os.walk(UPLOAD_DIR) if 'run_biobank.py' in files), None)
print('code folder:', CODE_DIR)
for rel in ['run_biobank.py', 'loaders/__init__.py', 'loaders/biovu.py', 'apoe_phewas/__init__.py',
            'apoe_phewas/config.py', 'apoe_phewas/phecode.py', 'apoe_phewas/cohort.py',
            'apoe_phewas/models.py', 'apoe_phewas/phewas.py', 'apoe_phewas/interaction.py',
            'apoe_phewas/descriptives.py']:
    ok = CODE_DIR is not None and os.path.isfile(os.path.join(CODE_DIR, rel))
    print('OK     ' if ok else 'MISSING', rel)   # MISSING = not uploaded, or imported as a notebook

# COMMAND ----------
# 3) Import the pipeline
sys.path.insert(0, CODE_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
import run_biobank, loaders.biovu
print('imports OK; results will go to', OUT_DIR)

# COMMAND ----------
# 4) Check that the source tables exist (all must print a row count)
from loaders.biovu import TABLES
TABLES = {k: v.replace('WORKSPACE_A.', f'{WORKSPACE_CATALOG}.') for k, v in TABLES.items()}
TABLES = {**TABLES, **TABLE_OVERRIDES}
for name, table in TABLES.items():
    if table is None:
        continue
    try:
        print(name, table, spark.sql(f'SELECT COUNT(*) AS n FROM {table}').collect()[0]['n'])
    except Exception as err:
        print('MISSING', name, table, str(err)[:120])

# COMMAND ----------
# 5) Load data (pandas on the driver)
from loaders.biovu import load
from run_biobank import run
persons, encounters, coded, pmap = load(spark, tables=TABLES)
if pmap is None:
    pmap = os.path.join(CODE_DIR, 'phecodeX_ICD_CM_map_flat.csv')
    print('PhecodeX map:', pmap, os.path.isfile(pmap))
print(len(persons), len(encounters), len(coded))

# COMMAND ----------
# 6) Stage 1. Part A and Part B together cover every Stage-1 analysis once.
STAGE1 = {
    'A': dict(analyses=('main', 'single'), subgroups=('Total', 'Male', 'Female', 'Black', 'Asian')),
    'B': dict(analyses=('table1', 'pcs', 'main', 'bands10'), subgroups=('White',)),
    '':  dict(analyses=('table1', 'pcs', 'main', 'bands10', 'single')),
}
run('biovu', persons, encounters, coded, pmap, OUT_DIR, part=PART, **STAGE1[PART])

# COMMAND ----------
# 7) Stage 2 — run in ONE environment only, after uploading into OUT_DIR: targets.csv (from
# summarise_all.py) and both main_results_A.csv and main_results_B.csv (the Firth step reads them)
run('biovu', persons, encounters, coded, pmap, OUT_DIR,
    analyses=('firth', 'interaction', 'continuous'), targets=f'{OUT_DIR}/targets.csv')
