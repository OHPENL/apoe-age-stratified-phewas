"""Run every analysis for one biobank and write summary statistics only.

Stage 1 (all biobanks): table1, pcs, main, bands10, single
Stage 2 (after summarise_all.py has written targets.csv): firth, interaction, continuous
"""
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

from apoe_phewas.cohort import (age_windows, build_cohort, lifetime_counts, subgroup,
                                window_counts)
from apoe_phewas.config import (AGE_BANDS, AGE_BANDS_10Y, AOU_MIN_COUNT, FDR_ALPHA, N_JOBS,
                                N_PCS, SUBGROUPS)
from apoe_phewas.descriptives import pc_variance, table1
from apoe_phewas.interaction import continuous_age_model, integrated_model, linear_age_model
from apoe_phewas.phecode import load_phecode_map, map_events, phecode_labels
from apoe_phewas.phewas import add_fdr, scan

ALL_ANALYSES = ('table1', 'pcs', 'main', 'bands10', 'single', 'firth', 'interaction', 'continuous',
                'linear')
COUNT_COLS = ('n_cases', 'n_controls', 'n_obs', 'n_persons', 'n_window')


def band_label(band) -> str:
    lo, hi = band
    return f'{lo}+' if hi >= 120 else f'{lo}-{hi - 1}'


def stratified_scan(cohort, bands, subgroups, single_stratum=False, cells=None,
                    method='auto', log=print, checkpoint=None) -> pd.DataFrame:
    """All PheCodes x age bands x subgroups. `cells` optionally limits PheCodes per
    (age_band, subgroup), e.g. to the significant ones for the Firth re-fit."""
    parts = []
    for band, win in age_windows(cohort, bands, single_stratum).items():
        label = band_label(band)
        counts = window_counts(cohort, *band)
        for g in subgroups:
            phecodes = None if cells is None else cells.get((label, g))
            if cells is not None and not phecodes:
                continue
            df, cov = subgroup(win, g, cohort.pc_cols)
            if df.empty:
                continue
            ck = checkpoint / f'{label}_{g}.csv' if checkpoint else None
            if ck is not None and ck.exists():      # finished in an earlier, interrupted run
                r = pd.read_csv(ck)
                log(f'  {label} {g}: loaded from checkpoint ({len(r)} PheCodes)')
            else:
                t0 = time.time()
                r = scan(df, cov, counts, phecodes=phecodes, method=method)
                log(f'  {label} {g}: N={len(df)}, tested={len(r)}, {time.time() - t0:.0f}s')
                if ck is not None:
                    r.to_csv(ck, index=False)
            if not r.empty:
                parts.append(r.assign(subgroup=g, age_band=label, n_window=len(df)))
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not res.empty:
        ok = res['p'].notna()
        res.loc[ok, 'fdr_global'] = add_fdr(res[ok].copy(), 'g')['g']   # one BH over every test
    return res


_CTX = {}   # cohort and counts shared with forked workers for the continuous-age models
CONTINUOUS_MAX_JOBS = 8   # each worker builds a large spline design matrix; bounds memory use


def _age_model_task(code):
    try:
        return _CTX['model'](_CTX['cohort'], _CTX['life'], code)
    except Exception as err:
        return pd.DataFrame([{'phecode': code, 'status': f'error: {str(err)[:80]}'}])


def _run_age_models(model, cohort, codes, log):
    """Per-PheCode continuous/linear age models, in parallel where fork is available."""
    _CTX.update(model=model, cohort=cohort, life=lifetime_counts(cohort, codes))
    if os.name != 'nt' and N_JOBS > 1:          # fork shares the cohort without copying
        try:
            with ProcessPoolExecutor(min(N_JOBS, CONTINUOUS_MAX_JOBS, len(codes)),
                                     mp_context=multiprocessing.get_context('fork')) as ex:
                return pd.concat(list(ex.map(_age_model_task, codes)), ignore_index=True)
        except (BrokenProcessPool, OSError) as err:
            log(f'  process pool failed ({type(err).__name__}); running serially')
    return pd.concat([_age_model_task(c) for c in codes], ignore_index=True)


def _ck(out: Path, name: str) -> Path:
    """Per-stratum checkpoints so an interrupted run resumes where it stopped."""
    d = out / '_checkpoints' / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _targets(targets, main, out):
    if targets is not None:
        return list(pd.read_csv(targets)['phecode']) if isinstance(targets, (str, Path)) else list(targets)
    if main is None:
        main = _read_main(out)
    sig = main[(main['subgroup'] == 'Total') & (main['fdr'] < FDR_ALPHA)]
    return sorted(sig['phecode'].unique())


def _read_main(out: Path) -> pd.DataFrame:
    """main_results.csv, or the parts (main_results_A.csv, ...) of a split run."""
    return pd.concat([pd.read_csv(f) for f in sorted(out.glob('main_results*.csv'))], ignore_index=True)


def run(biobank, persons, encounters, coded_events, phecode_map, out_dir,
        analyses=ALL_ANALYSES, targets=None, subgroups=SUBGROUPS, n_pcs=N_PCS, log=print,
        part: str = ''):
    """part: label for a run split across environments (e.g. 'A', 'B'); added to file names."""
    suffix = f'_{part}' if part else ''
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suppress = AOU_MIN_COUNT if biobank == 'aou' else 0
    pmap = load_phecode_map(phecode_map)
    labels = phecode_labels(pmap)

    def save(df, name):
        df = df.copy()
        if suppress:                                   # All of Us dissemination policy
            for col in set(COUNT_COLS) & set(df.columns):
                df[col] = df[col].where(~df[col].between(1, suppress - 1), np.nan)
        if 'phecode' in df.columns and 'phecode_string' not in df.columns:
            df = df.merge(labels, on='phecode', how='left')
        df.insert(0, 'biobank', biobank)
        df.to_csv(out / f'{name}{suffix}.csv', index=False)
        log(f'wrote {name}{suffix}.csv ({len(df)} rows)')

    log(f'[{biobank}] building cohort')
    events = map_events(coded_events, pmap, truncate=(biobank == 'ukb'))
    cohort = build_cohort(persons, encounters, events, n_pcs)
    save(pd.DataFrame([cohort.flow]), 'cohort_flow')

    if 'table1' in analyses:
        save(table1(cohort, suppress), 'table1')
    if 'pcs' in analyses:
        save(pc_variance(cohort), 'pc_variance')

    main = None
    if 'main' in analyses:
        log('main analysis')
        main = stratified_scan(cohort, AGE_BANDS, subgroups, log=log, checkpoint=_ck(out, 'main'))
        save(main, 'main_results')
    if 'bands10' in analyses:
        log('10-year age bands (Total)')
        save(stratified_scan(cohort, AGE_BANDS_10Y, ['Total'], log=log,
                             checkpoint=_ck(out, 'bands10')), 'bands10_results')
    if 'single' in analyses:
        log('single-stratum sensitivity (Total)')
        save(stratified_scan(cohort, AGE_BANDS, ['Total'], single_stratum=True, log=log,
                             checkpoint=_ck(out, 'single')),
             'single_stratum_results')

    if 'firth' in analyses:
        log('Firth re-fit of significant associations')
        m = main if main is not None else _read_main(out)
        sig = m[m['fdr'] < FDR_ALPHA]
        cells = sig.groupby(['age_band', 'subgroup'])['phecode'].apply(list).to_dict()
        firth = stratified_scan(cohort, AGE_BANDS, subgroups, cells=cells, method='firth', log=log)
        save(firth.drop(columns=['fdr', 'fdr_global'], errors='ignore'), 'firth_results')

    if {'interaction', 'continuous', 'linear'} & set(analyses):
        codes = _targets(targets, main, out)
        log(f'{len(codes)} target PheCodes for interaction models')
        if 'interaction' in analyses:
            windows = age_windows(cohort, AGE_BANDS)
            counts = {b: window_counts(cohort, *b, phecodes=codes) for b in windows}
            rows = []
            for code in codes:
                try:
                    rows.append(integrated_model(windows, counts, code, cohort.pc_cols))
                except Exception as err:                      # keep going; record the failure
                    rows.append({'phecode': code, 'status': f'error: {str(err)[:80]}'})
            save(pd.DataFrame(rows), 'interaction_results')
        if 'continuous' in analyses:
            save(_run_age_models(continuous_age_model, cohort, codes, log), 'continuous_age_results')
        if 'linear' in analyses:
            save(_run_age_models(linear_age_model, cohort, codes, log), 'linear_age_results')
    log('done')
    return cohort
