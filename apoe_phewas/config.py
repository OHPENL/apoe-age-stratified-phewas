"""Analysis settings shared by all three biobanks.

Every pre-specified choice lives here so that the Methods text, the reviewer
responses and the code can be checked against one place.
"""

# Attained-age strata (lower bound inclusive, upper bound exclusive).
AGE_BANDS = ((18, 45), (45, 60), (60, 75), (75, 120))
# Narrower strata for the sensitivity analysis requested by Reviewer 4 (comment 1).
AGE_BANDS_10Y = ((18, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 120))

ADULT_AGE = 18
MIN_ADULT_ENCOUNTER_DATES = 2   # distinct dated records at age >= 18
MIN_FOLLOWUP_DAYS = 365         # first to last adult record

N_PCS = 16                      # harmonised across biobanks (Reviewer 1, comment 3)

# Case definition: PheCode recorded on >= 2 distinct dates within the age window.
MIN_CASE_DATES = 2
# Individuals with exactly one dated record are excluded from that PheCode's
# controls rather than counted as controls (Reviewer 5, comment 1).
EXCLUDE_SINGLE_MENTION = True
# 'window': diagnoses dated inside the age window (age-specific profile).
# 'cumulative': diagnoses dated from age 18 up to the end of the window.
CASE_SCOPE = 'window'

# Pre-specified: a PheCode is tested in a stratum only if it has at least this many cases.
MIN_CASES = 10
FDR_ALPHA = 0.05

# Worker processes for the PheCode loop (results are identical to a serial run).
import os as _os
N_JOBS = max(1, (_os.cpu_count() or 2) - 2)

GENO_TO_E4 = {'E3_E3': 0, 'E3_E4': 1, 'E4_E4': 2}   # epsilon-2 carriers are excluded
RACES = ('White', 'Black', 'Asian')                  # race-stratified groups
RACE_LEVELS = ('White', 'Black', 'Asian', 'Other', 'Unknown')
SUBGROUPS = ('Total', 'Male', 'Female', 'White', 'Black', 'Asian')

# US diagnosis codes switched from ICD-9-CM to ICD-10-CM on this date.
ICD10CM_START = '2015-10-01'

# All of Us Data and Statistics Dissemination Policy: no counts 1-19 may leave the Workbench.
AOU_MIN_COUNT = 20

# Continuous-age model (Reviewer 4, comment 1).
SPLINE_DF = 4
CONTINUOUS_AGE_GRID = tuple(range(25, 90, 5))
