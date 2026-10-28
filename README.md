# Age-stratified phenome-wide association study of APOE ε4

Analysis code for *An Age-Stratified Phenome-Wide Association Study of APOE ε4: Age-, Sex-, and Race-Specific
Associations Across Three Independent Biobanks* (Jeong, Feng, Hohman, Guo, Chen; eBioMedicine, under review).

The same code and pre-specified settings are run separately in VUMC BioVU (Databricks), the All of Us Researcher
Workbench, and UK Biobank. Individual-level data never leave each platform; only summary statistics are exported.
No participant data are included in this repository.

## Contents
```
apoe_phewas/      config.py (all pre-specified settings), phecode.py, cohort.py, models.py, phewas.py,
                  interaction.py, descriptives.py
loaders/          ukb.py (local files), biovu.py (Databricks/Spark), aou.py (Researcher Workbench/BigQuery)
run_biobank.py    runs every analysis for one biobank
run_ukb.py        command-line entry point for UK Biobank
notebooks/        biovu_databricks.py, aou_workbench.py (cell-by-cell platform notebooks)
summarise_all.py  combines the three biobanks' summary statistics
tools/            build_ukb_conditions.py (Read v2/CTV3 to ICD-10 conversion), make_figure1.py
tests/            smoke_test.py (end-to-end run on synthetic data)
```

## Analysis summary
- Cohort: ε3/ε3, ε3/ε4, ε4/ε4 carriers (ε2 carriers excluded); ≥2 adult EHR dates spanning ≥1 year.
- Attained-age strata 18–44, 45–59, 60–74, ≥75 years; case = PheCode (PhecodeX) on ≥2 dates within the window;
  individuals with a single record are excluded from that PheCode's analysis.
- Logistic regression of case status on ε4 allele count, adjusted for age and follow-up in the window, sex, race,
  and 16 ancestry principal components; unpenalised maximum likelihood with Firth's correction on separation;
  PheCodes with ≥10 cases; Benjamini–Hochberg FDR within biobank × stratum × analysis group.
- Joint ε4 × age, ε4 × sex, ε4 × race models with participant-clustered SEs; continuous-age (spline and linear)
  models; sensitivity analyses (Firth, one stratum per participant, 10-year bands, study-wide FDR).

## Running
```
pip install -r requirements.txt
python tests/smoke_test.py                       # synthetic data; prints SMOKE TEST PASSED

# UK Biobank (stage 1: descriptives, main scan, 10-year bands, single stratum)
python tools/build_ukb_conditions.py --ukb-dir /path/to/ukb
python run_ukb.py --data-dir /path/to/ukb --out out/ukb --stage 1
# BioVU and All of Us: run notebooks/biovu_databricks.py and notebooks/aou_workbench.py cell by cell
python summarise_all.py --biovu out/biovu --aou out/aou --ukb out/ukb --out out/combined   # writes targets.csv
# stage 2 in each biobank (Firth, joint interaction, continuous-age models), then summarise_all.py again
python run_ukb.py --data-dir /path/to/ukb --out out/ukb --stage 2 --targets out/combined/targets.csv
```

## Inputs not distributed here
- The PhecodeX ICD-9-CM/ICD-10-CM map (Shuey et al., Bioinformatics 2023) and, for UK Biobank, the UK Biobank
  lookup tables (`all_lkps_maps_v4.xlsx`), both available from their providers.
- Biobank data, available to approved researchers from VUMC BioVU, the All of Us Research Program, and UK Biobank.
  Databricks table names in `loaders/biovu.py` are placeholders (`CATALOG.SCHEMA.table`).
- All of Us exports suppress counts of 1–19, following the Data and Statistics Dissemination Policy.

## Citation
See `CITATION.cff`. Please cite the archived release (Zenodo DOI) and the article.

## Licence
MIT (see `LICENSE`).
