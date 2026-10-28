"""Figure 1 (study workflow) with the revision cohort numbers -> out/figures/Figure1.png/.svg"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'out'
plt.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'stix'})

BIOBANKS = [('biovu', 'VUMC BioVU', 'ICD-9-CM/ICD-10-CM → PheCodes'),
            ('aou', 'All of Us Research Program', 'ICD-9-CM/ICD-10-CM → PheCodes'),
            ('ukb', 'UK Biobank', 'Read v2/CTV3 → ICD-10 → PheCodes')]


def numbers(b):
    flow = pd.concat([pd.read_csv(f) for f in (O / b).glob('cohort_flow*.csv')]).iloc[0]
    t1 = pd.concat([pd.read_csv(f) for f in (O / b).glob('table1*.csv')])
    n = t1[t1['variable'] == 'N'].iloc[0]
    main = pd.concat([pd.read_csv(f) for f in (O / b).glob('main_results*.csv')])
    tested = main.loc[main['subgroup'] == 'Total', 'phecode'].nunique()
    e3, e4 = int(float(n['e4_0'])), int(float(n['e4_1'])) + int(float(n['e4_2']))
    return {'genotyped': int(flow['genotyped']), 'e2': int(flow['epsilon2_or_unknown_genotype_excluded']),
            'criteria': int(flow['genotyped'] - flow['epsilon2_or_unknown_genotype_excluded'] - flow['eligible']),
            'n': int(flow['eligible']), 'e3': e3, 'e4': e4, 'tested': tested}


def box(ax, x, y, w, h, title, lines, title_size=13, size=11, align='center'):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.5))
    tx = x + w / 2 if align == 'center' else x + 0.012
    ax.text(tx, y + h - 0.02, title, ha=align, va='top', fontsize=title_size, fontweight='bold')
    for i, line in enumerate(lines):
        ax.text(tx, y + h - 0.056 - i * 0.03, line, ha=align, va='top', fontsize=size)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black'))


def main():
    fig = plt.figure(figsize=(13.5, 8.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    w, gap = 0.19, 0.325
    for k, (b, name, coding) in enumerate(BIOBANKS):
        n = numbers(b)
        x = 0.02 + k * gap
        box(ax, x, 0.88, w, 0.09, name, [f'Genotyped: n = {n["genotyped"]:,}'])
        box(ax, x + w + 0.008, 0.79, 0.122, 0.11, 'Exclusion',
            [f'ε2 carriers: {n["e2"]:,}', f'Exclusion criteria$^{{*}}$: {n["criteria"]:,}'],
            title_size=11, size=9.5, align='left')
        box(ax, x, 0.66, w, 0.16, 'Analytic cohort',
            [f'n = {n["n"]:,}', f'ε3/ε3 reference: {n["e3"]:,}', f'ε4 carriers$^{{\\dagger}}$: {n["e4"]:,}'])
        box(ax, x, 0.42, w, 0.19, 'Phenotyping',
            [coding, 'case: PheCode on ≥2 dates', 'within the age window',
             f'PheCodes tested: {n["tested"]:,}'], size=10.5)
        cx = x + w / 2
        arrow(ax, cx, 0.88, cx, 0.82)
        ax.plot([cx, x + w + 0.004], [0.85, 0.85], color='black', lw=1.5)
        arrow(ax, x + w + 0.0, 0.85, x + w + 0.008, 0.85)
        arrow(ax, cx, 0.66, cx, 0.61)
        ax.plot([cx, cx], [0.42, 0.395], color='black', lw=1.5)
    ax.plot([0.02 + w / 2, 0.02 + 2 * gap + w / 2], [0.395, 0.395], color='black', lw=1.5)
    arrow(ax, 0.5, 0.395, 0.5, 0.37)
    box(ax, 0.02, 0.2, 0.96, 0.17, 'Age-stratified PheWAS (each biobank analysed separately)', [
        'Logistic regression (maximum likelihood; Firth if separation) · additive APOE ε4 dose (ε3/ε3 = 0, ε3/ε4 = 1, ε4/ε4 = 2)',
        'Attained-age strata 18–44, 45–59, 60–74, ≥75 y · adjusted for age, sex, EHR follow-up, 16 PCs · ≥10 cases',
        'Overall and by sex and self-reported race · FDR-corrected p < 0.05 with OR > 1 (per biobank, stratum, and group)',
    ], size=11)
    arrow(ax, 0.5, 0.2, 0.5, 0.175)
    box(ax, 0.02, 0.04, 0.96, 0.135, 'Replication, effect modification, and sensitivity analyses', [
        'Replication across ≥2 and all three biobanks · known vs not catalogued (DisGeNET)',
        'Joint ε4 × age, × sex, × race model (person-clustered SEs) · continuous-age models',
        'Firth regression · one stratum per person · 10-year age bands · study-wide FDR',
    ], size=11)
    fig.text(0.02, 0.021, '$^{*}$Age < 18 years at all records, < 2 adult EHR dates, < 1 year of adult EHR follow-up, '
             'or missing sex or ancestry principal components.', fontsize=9.5, style='italic')
    fig.text(0.02, 0.004, '$^{\\dagger}$ε4 carriers = ε3/ε4 + ε4/ε4.', fontsize=9.5, style='italic')
    out = O / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'Figure1.png', dpi=300)
    fig.savefig(out / 'Figure1.svg')
    print('saved', out / 'Figure1.png')


if __name__ == '__main__':
    main()
