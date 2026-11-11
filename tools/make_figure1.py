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


def small_box(ax, x, y, w, h, lines, size=10.5):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.2))
    step = h / (len(lines) + 1)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - step * (i + 1), line, ha='center', va='center', fontsize=size,
                fontweight='bold' if i == 0 and len(lines) > 1 else 'normal')


def analysis_section(ax):
    """Main (overall) analysis, then the stratified analyses and effect-modification models that build on it."""
    arrow(ax, 0.5, 0.395, 0.5, 0.37)
    box(ax, 0.05, 0.195, 0.90, 0.175, 'Main analysis: age-stratified PheWAS in all participants (overall analysis)', [])
    ax.text(0.5, 0.31, 'Each biobank analyzed separately, and results compared across biobanks',
            ha='center', va='center', fontsize=11)
    for k, band in enumerate(['18–44 years', '45–59 years', '60–74 years', '≥75 years']):
        small_box(ax, 0.17 + k * 0.17, 0.258, 0.15, 0.036, [band], size=11)
    ax.text(0.5, 0.228, 'Logistic regression of each PheCode on ε4 allele count, adjusted for age, sex, race, EHR '
            'follow-up, and 16 PCs · FDR p < 0.05', ha='center', va='center', fontsize=10.5)
    follow = [(0.27, 'Sex-stratified and race-stratified analyses', ['Sex: males, females', 'Race: White, Black, Asian']),
              (0.73, 'Effect modification', ['Joint model across the four strata', 'ε4 × age, ε4 × sex, ε4 × race'])]
    for cx, title, lines in follow:
        arrow(ax, cx, 0.195, cx, 0.15)
        box(ax, cx - 0.22, 0.015, 0.44, 0.135, title, lines, title_size=12, size=10.5)


def main():
    fig = plt.figure(figsize=(13.5, 8.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 1)
    ax.axis('off')
    w, gap = 0.19, 0.325
    for k, (b, name, coding) in enumerate(BIOBANKS):
        n = numbers(b)
        x = 0.02 + k * gap
        box(ax, x, 0.88, w, 0.09, name, [f'Genotyped: n = {n["genotyped"]:,}'])
        box(ax, x + w + 0.008, 0.79, 0.122, 0.11, 'Exclusion',
            [f'ε2 carriers: {n["e2"]:,}', f'Other exclusions: {n["criteria"]:,}'],
            title_size=11, size=9.5, align='left')
        box(ax, x, 0.66, w, 0.16, 'Analytic cohort',
            [f'n = {n["n"]:,}', f'ε3/ε3 reference: {n["e3"]:,}', f'ε4 carriers (ε3/ε4, ε4/ε4): {n["e4"]:,}'])
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
    analysis_section(ax)
    out = O / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'Figure1.png', dpi=300)
    fig.savefig(out / 'Figure1.svg')
    print('saved', out / 'Figure1.png')


if __name__ == '__main__':
    main()
