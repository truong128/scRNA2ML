"""Publication tables (TSV + one Excel workbook) and figures (PDF + PNG) built only from saved pipeline outputs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_curve

from . import graphical

CITATION = {'GSE117570': 'Song et al. 2019', 'GSE131907': 'Kim et al. 2020', 'GSE154826': 'Leader et al. 2021',
            'GSE123902': 'Laughney et al. 2020', 'GSE198099': 'GEO GSE198099', 'E-MTAB-13526': 'De Zuani et al. 2024'}
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']
TISSUE = {'Normal': '#4C9F70', 'Tumor': '#C0392B'}
SCHEME_LABEL = {'loco': 'Leave-one-cohort-out', 'loco_centered': 'LOCO, cohort-centred', 'pooled': 'Pooled donor CV'}
MM = 1 / 25.4


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'mathtext.fontset': 'stix', 'font.size': 10,
                         'axes.titlesize': 10.5, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
                         'legend.title_fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 300, 'figure.dpi': 150})


def panel(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')


def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), bbox_inches='tight', dpi=300)
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def fmt_ci(v, lo, hi, digits=2):
    if pd.isna(v):
        return 'NA'
    return f'{v:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})' if pd.notna(lo) else f'{v:.{digits}f}'


class Results:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg['study']['results_dir'])
        self.data = Path(cfg['study']['data_dir'])

    def units(self, comp):
        return pd.read_csv(self.root / 'features' / comp / 'units.tsv', sep='\t')

    def X(self, comp):
        return pd.read_csv(self.root / 'features' / comp / 'X.tsv.gz', sep='\t')

    def ml_dir(self, comp, scheme):
        return self.root / 'ml' / comp / scheme.replace('::', '_')

    def schemes(self, comp):
        base = self.root / 'ml' / comp
        return sorted(p.name for p in base.iterdir() if (p / 'metrics.tsv').exists()) if base.exists() else []

    def metrics(self, comp, scheme):
        return pd.read_csv(self.ml_dir(comp, scheme) / 'metrics.tsv', sep='\t')

    def oof(self, comp, scheme):
        return pd.read_csv(self.ml_dir(comp, scheme) / 'oof_predictions.tsv', sep='\t')

    def perm(self, comp, scheme):
        p = self.ml_dir(comp, scheme) / 'permutation_test.json'
        return json.loads(p.read_text()) if p.exists() else None

    def analysis(self, comp, name):
        return pd.read_csv(self.root / 'analysis' / comp / name, sep='\t')


def n_candidates(R, comp, scheme):
    return int((~R.metrics(comp, scheme).model.isin(['nested_algorithm_selection', 'dummy_prior'])).sum())


def scheme_label(s):
    if s.startswith('within_'):
        return f'Within {s[len("within_"):]}'
    return SCHEME_LABEL.get(s, s)


# ----------------------------------------------------------------------------- tables
def table_cohorts(R):
    libs = pd.read_csv(R.data / 'prepared' / 'libraries.tsv', sep='\t')
    qc = pd.read_csv(R.root / 'seurat' / 'qc_audit.tsv', sep='\t')
    rows = []
    for cohort in libs.cohort.unique():
        q = qc[qc.cohort == cohort]
        row = {'Cohort': cohort, 'Reference': CITATION.get(cohort, ''), 'Chemistry': '; '.join(sorted(q.chemistry.astype(str).unique())),
               'Cell fractions': '; '.join(sorted(q.enrichment.unique())), 'Libraries (eligible)': len(q),
               'Libraries passing QC': int((q.status == 'complete').sum()), 'Cells passing QC': int(q.retained_cells.fillna(0).sum())}
        for comp in R.cfg['features']['compartments']:
            u = R.units(comp)
            u = u[u.cohort == cohort].drop_duplicates('donor_id')
            row[f'Paired donors ({comp})'] = len(u)
            if comp == 'immune':
                row['Histology (immune donors)'] = ', '.join(f'{k} {v}' for k, v in u.histology.value_counts().items())
                row['Fraction used (immune donors)'] = ', '.join(f'{k} {v}' for k, v in u.enrichment.value_counts().items())
        rows.append(row)
    t = pd.DataFrame(rows)
    total = {'Cohort': 'Total', 'Libraries (eligible)': t['Libraries (eligible)'].sum(), 'Libraries passing QC': t['Libraries passing QC'].sum(),
             'Cells passing QC': t['Cells passing QC'].sum()}
    for comp in R.cfg['features']['compartments']:
        total[f'Paired donors ({comp})'] = t[f'Paired donors ({comp})'].sum()
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True)


def table_performance(R):
    rows = []
    for comp in R.cfg['features']['compartments']:
        for scheme in R.schemes(comp):
            m = R.metrics(comp, scheme).set_index('model')
            perm = R.perm(comp, scheme)
            for model in ('logistic_regression', 'nested_algorithm_selection'):
                if model not in m.index or not m.loc[model, 'complete_oof']:
                    continue
                r = m.loc[model]
                rows.append({'Compartment': comp, 'Validation': scheme_label(scheme), 'Model': 'Logistic regression (prespecified)' if model == 'logistic_regression' else f'Nested algorithm selection ({n_candidates(R, comp, scheme)} candidates)',
                             'Donors': int(r.n_donors), 'Balanced accuracy (95% CI)': fmt_ci(r.balanced_accuracy, r.balanced_accuracy_ci_low, r.balanced_accuracy_ci_high),
                             'AUROC (95% CI)': fmt_ci(r.get('auroc'), r.get('auroc_ci_low'), r.get('auroc_ci_high')) if model == 'logistic_regression' else 'NA',
                             'Mean per-fold AUROC': f"{r['auroc_mean_over_folds']:.2f}" if pd.notna(r.get('auroc_mean_over_folds')) else 'NA',
                             'Paired concordance': f"{r['paired_concordance']:.2f}" if pd.notna(r.get('paired_concordance')) else 'NA',
                             'Sensitivity': f'{r.sensitivity_tumor:.2f}', 'Specificity': f'{r.specificity_tumor:.2f}',
                             'Permutation P (BA)': f"{perm['p_balanced_accuracy']:.2g}" if perm and model == 'logistic_regression' else ''})
    return pd.DataFrame(rows)


def write_tables(R, out):
    out.mkdir(parents=True, exist_ok=True)
    tables = {'Table1_cohorts': table_cohorts(R), 'Table2_performance': table_performance(R)}
    tables['S1_exclusions'] = pd.read_csv(R.data / 'prepared' / 'exclusions.tsv', sep='\t')
    tables['S2_library_QC'] = pd.read_csv(R.root / 'seurat' / 'qc_audit.tsv', sep='\t').drop(columns=['path'])
    for comp in R.cfg['features']['compartments']:
        tables[f'S3_units_{comp}'] = R.units(comp)
        ue = pd.read_csv(R.root / 'features' / comp / 'unit_exclusions.tsv', sep='\t')
        if len(ue):
            tables[f'S4_unit_exclusions_{comp}'] = ue
        for scheme in R.schemes(comp):
            tables[f'S5_{comp}_{scheme}_all_models'] = R.metrics(comp, scheme).sort_values('balanced_accuracy', ascending=False)
        tables[f'S6_{comp}_meta_analysis'] = R.analysis(comp, 'paired_meta_analysis.tsv')
        tables[f'S7_{comp}_ablation'] = R.analysis(comp, 'feature_set_ablation_loco.tsv')
        tables[f'S8_{comp}_subgroups'] = R.analysis(comp, 'subgroups_loco.tsv')
        tables[f'S9_{comp}_stability'] = R.analysis(comp, 'selection_stability_loco.tsv')
        tables[f'S10_{comp}_final_coef'] = pd.read_csv(R.root / 'ml' / comp / 'final_model_coefficients.tsv', sep='\t')
    for name, t in tables.items():
        t.to_csv(out / f'{name}.tsv', sep='\t', index=False)
    with pd.ExcelWriter(out / 'tables.xlsx') as xl:
        for name, t in tables.items():
            t.to_excel(xl, sheet_name=name[:31], index=False)
    return tables


# ----------------------------------------------------------------------------- figures
def cohort_colors(units):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(units.cohort.unique()))}


def fig_data(R, out, comp='immune'):
    units, X = R.units(comp), R.X(comp)
    qc = pd.read_csv(R.root / 'seurat' / 'qc_audit.tsv', sep='\t')
    colors = cohort_colors(units)
    cohorts = sorted(units.cohort.unique())
    fig = plt.figure(figsize=(210 * MM, 190 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    d = units.drop_duplicates('donor_id').groupby(['cohort', 'enrichment']).size().unstack(fill_value=0).reindex(cohorts)
    left = np.zeros(len(d))
    for i, col in enumerate(d.columns):
        ax.barh(d.index, d[col], left=left, color=['#555555', '#BBBBBB', '#888888'][i % 3], label=col, height=0.7)
        left += d[col].to_numpy()
    for i, v in enumerate(left):
        ax.text(v + 0.4, i, str(int(v)), va='center', fontsize=9)
    ax.set_xlabel('Paired donors')
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1)
    ax.invert_yaxis()
    panel(ax, 'a')
    ax = fig.add_subplot(gs[0, 1])
    q = qc[qc.status == 'complete']
    for i, c in enumerate(cohorts):
        v = q[q.cohort == c].retained_cells.to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.2, .2, len(v)), v, s=6, color=colors[c], alpha=.8, lw=0)
        ax.hlines(np.median(v), i - .3, i + .3, color='k', lw=.8)
    ax.set_yscale('log')
    ax.set_xticks(range(len(cohorts)), cohorts, rotation=45, ha='right')
    ax.set_ylabel('Cells passing QC per library')
    panel(ax, 'b')
    ax = fig.add_subplot(gs[0, 2])
    q2 = q.assign(immune_fraction=q.immune_cells / q.retained_cells)
    for i, c in enumerate(cohorts):
        v = q2[q2.cohort == c]
        for j, (frac, mk) in enumerate([('CD45+', 'o'), ('unsorted', '^')]):
            s = v[v.enrichment == frac].immune_fraction.to_numpy()
            ax.scatter(np.full(len(s), i) + (j - .5) * .3, s, s=7, color=colors[c], marker=mk, lw=0, alpha=.85)
    ax.set_xticks(range(len(cohorts)), cohorts, rotation=45, ha='right')
    ax.set_ylabel('Immune fraction of QC cells')
    ax.set_ylim(0, 1.02)
    ax.legend([Line2D([], [], marker='o', ls='', color='grey', ms=4), Line2D([], [], marker='^', ls='', color='grey', ms=4)], ['CD45+', 'unsorted'], frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3)
    panel(ax, 'c')
    ax = fig.add_subplot(gs[1, 0:2])
    props = [c for c in X if c.startswith('prop::')]
    names = [p.split('::')[1] for p in props]
    mean = X[props].groupby([units.cohort, units.tissue]).mean()
    order = [(c, t) for c in cohorts for t in ('Normal', 'Tumor')]
    mean = mean.reindex(order)
    cmap = plt.get_cmap('tab10')
    bottom = np.zeros(len(mean))
    xs = np.arange(len(mean)) + np.repeat(np.arange(len(cohorts)) * .6, 2)
    for i, (p, n) in enumerate(zip(props, names)):
        ax.bar(xs, mean[p], bottom=bottom, color=cmap(i % 10), width=.9, label=n)
        bottom += mean[p].to_numpy()
    ax.set_xticks(xs, ['N' if t == 'Normal' else 'T' for _, t in order])
    for i, c in enumerate(cohorts):
        ax.text(xs[2 * i] + .5, -0.09, c, ha='right', va='top', rotation=30, rotation_mode='anchor', fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_ylabel(f'Mean proportion ({comp} compartment)')
    ax.legend(frameon=False, ncol=len(props), bbox_to_anchor=(0.5, 1.0), loc='lower center', handlelength=1, columnspacing=1)
    ax.set_ylim(0, 1)
    panel(ax, 'd')
    ax = fig.add_subplot(gs[1, 2])
    meta = R.analysis(comp, 'paired_meta_analysis.tsv')
    per = R.analysis(comp, 'paired_differences_per_cohort.tsv')
    mp = meta[meta.type == 'proportion'].set_index('feature').reindex(props).dropna(subset=['pooled_difference'])
    ys = np.arange(len(mp))
    for i, (f, r) in enumerate(mp.iterrows()):
        pc = per[per.feature == f]
        for _, s in pc.iterrows():
            ax.scatter(100 * s.mean_paired_difference, i + np.random.default_rng(i).uniform(-.15, .15), s=6, color=colors[s.cohort], lw=0, alpha=.8)
        ax.errorbar(100 * r.pooled_difference, i, xerr=100 * 1.96 * r.se, fmt='D', color='k', ms=3, lw=.8, capsize=1.5)
        ax.text(1.02, i, f"q={r.fdr:.1g}", transform=ax.get_yaxis_transform(), fontsize=8.5, va='center')
    ax.axvline(0, color='grey', lw=.5, ls='--')
    ax.set_yticks(ys, [f.split('::')[1] for f in mp.index])
    ax.invert_yaxis()
    ax.set_xlabel('Tumour − normal (percentage points)')
    panel(ax, 'e')
    handles = [Line2D([], [], marker='o', ls='', color=colors[c], ms=3, label=c) for c in cohorts]
    fig.legend(handles=handles, loc='lower center', ncol=len(cohorts), frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, out / 'Figure2_cohorts_qc_composition')


def fig_performance(R, out, comp='immune'):
    colors = cohort_colors(R.units(comp))
    fig = plt.figure(figsize=(215 * MM, 215 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.62, width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, 0])
    pf = pd.read_csv(R.ml_dir(comp, 'loco') / 'metrics_per_fold.tsv', sep='\t')
    m = R.metrics(comp, 'loco')
    m = m[m.complete_oof.astype(bool)]
    top = m[~m.model.isin(['nested_algorithm_selection', 'dummy_prior'])].sort_values('balanced_accuracy', ascending=False).model.tolist()
    rows = ['logistic_regression', 'nested_algorithm_selection'] + [t for t in top if t != 'logistic_regression'][:14] + ['dummy_prior']
    rows = [r for r in rows if r in set(pf.model)]
    heat = pf.pivot_table(index='model', columns='test', values='balanced_accuracy').reindex(rows)
    im = ax.imshow(heat.to_numpy(), cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(rows)), [r.replace('_', ' ') for r in rows])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5, color='white' if abs(v - .5) > .3 else 'black')
    overall = m.set_index('model').reindex(rows).balanced_accuracy
    for i, v in enumerate(overall):
        ax.text(heat.shape[1] - .35, i, f'{v:.2f}', va='center', fontsize=8.5, fontweight='bold')
    ax.text(heat.shape[1] - .35, -1, 'All', fontsize=9, fontweight='bold')
    ax.set_xlabel('Held-out cohort')
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.12, label='Balanced accuracy')
    ax.spines[:].set_visible(False)
    panel(ax, 'a')
    ax = fig.add_subplot(gs[0, 1])
    oof = R.oof(comp, 'loco')
    p = oof[oof.model == 'logistic_regression']
    for c, d in p.groupby('cohort'):
        fpr, tpr, _ = roc_curve(d.truth, d.score)
        ax.plot(fpr, tpr, color=colors[c], lw=1, label=f'{c} (n={d.donor.nunique()})')
    fpr, tpr, _ = roc_curve(p.truth, p.score)
    r = m.set_index('model').loc['logistic_regression']
    ax.plot(fpr, tpr, color='k', lw=1.6, label=f"Pooled AUROC {r.auroc:.2f}")
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=.5)
    ax.set(xlabel='1 − specificity', ylabel='Sensitivity', title='Prespecified logistic model, LOCO')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_aspect('equal')
    panel(ax, 'b')
    ax = fig.add_subplot(gs[1, 0])
    entries = []
    for c2 in R.cfg['features']['compartments']:
        for s in R.schemes(c2):
            mm = R.metrics(c2, s).set_index('model')
            for model, mk in (('logistic_regression', 'o'), ('nested_algorithm_selection', 's')):
                if model in mm.index and mm.loc[model, 'complete_oof']:
                    rr = mm.loc[model]
                    entries.append((f'{c2}: {scheme_label(s)}', model, rr.balanced_accuracy, rr.balanced_accuracy_ci_low, rr.balanced_accuracy_ci_high, int(rr.n_donors), mk))
    labels = list(dict.fromkeys(e[0] for e in entries))
    for e in entries:
        y = labels.index(e[0]) + (-.15 if e[1] == 'logistic_regression' else .15)
        ax.errorbar(e[2], y, xerr=[[e[2] - e[3]], [e[4] - e[2]]], fmt=e[6], color='#C0392B' if e[1] == 'logistic_regression' else '#2C3E50', ms=3.5, lw=.8, capsize=1.5)
    ax.set_yticks(range(len(labels)), [f'{l} (n={next(e[5] for e in entries if e[0] == l)})' for l in labels])
    ax.axvline(.5, ls='--', color='grey', lw=.5)
    ax.set_xlim(0.3, 1.02)
    ax.invert_yaxis()
    ax.set_xlabel('Balanced accuracy (95% donor-bootstrap CI)')
    ax.legend([Line2D([], [], marker='o', color='#C0392B', ls=''), Line2D([], [], marker='s', color='#2C3E50', ls='')], ['Logistic (prespecified)', f"Nested selection ({n_candidates(R, comp, 'loco')} candidates)"], frameon=False, loc='upper left', fontsize=8.5)
    panel(ax, 'c')
    ax = fig.add_subplot(gs[1, 1])
    ablation_panel(R, ax, comp)
    panel(ax, 'd')
    save(fig, out / 'Figure3_model_performance')


def ablation_panel(R, ax, comp):
    ab = R.analysis(comp, 'feature_set_ablation_loco.tsv')
    sets = ab[~ab.feature_set.str.startswith('single::')].sort_values('auroc', ascending=False)
    single = ab[ab.feature_set.str.startswith('single::')].sort_values('auroc', ascending=False)
    rows = [(s.feature_set.replace('_', ' '), s.auroc, '#C0392B') for s in sets.itertuples()]
    rows += [(s.feature_set.split('::')[-1], s.auroc, '#7F8C8D') for s in single.itertuples()]
    y = np.arange(len(rows)) + np.where(np.arange(len(rows)) >= len(sets), .6, 0)
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=.75)
    for yi, r in zip(y, rows):
        ax.text(r[1] + .015, yi, f'{r[1]:.2f}', va='center', fontsize=8.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(.5, ls='--', color='grey', lw=.5)
    ax.set_xlim(.3, 1.08)
    ax.set_xlabel('Leave-one-cohort-out AUROC')
    ax.legend([Line2D([], [], color='#C0392B', lw=6), Line2D([], [], color='#7F8C8D', lw=6)],
              ['Feature set (nested logistic)', 'Single cell-type proportion'], frameon=False, fontsize=8.5,
              loc='lower center', bbox_to_anchor=(.45, 1.0), ncol=1)


def permutation_panel(R, ax):
    tests = [(c2, s) for c2, ss in R.cfg['ml'].get('permutation_schemes', {}).items() for s in ss
             if (R.ml_dir(c2, s) / 'permutation_null.tsv').exists()]
    rng = np.random.default_rng(0)
    for i, (c2, s) in enumerate(tests):
        null = pd.read_csv(R.ml_dir(c2, s) / 'permutation_null.tsv', sep='\t').auroc
        perm = R.perm(c2, s)
        cut = float(np.quantile(null, .95))
        ax.scatter(null, i + rng.uniform(-.17, .17, len(null)), s=4, color='#AAAAAA', lw=0, rasterized=True)
        ax.plot([cut, cut], [i - .3, i + .3], color='k', lw=1.2)
        ax.scatter(perm['observed_auroc'], i, marker='D', s=30, color='#C0392B', zorder=3)
        ax.text(perm['observed_auroc'], i - .36, f"{perm['observed_auroc']:.2f}", color='#C0392B', fontsize=8.5, ha='center', va='bottom')
        ax.text(cut, i + .36, f'{cut:.2f}', fontsize=8.5, ha='center', va='top')
    ax.set_yticks(range(len(tests)), [f'{c2}\n{scheme_label(s)}' for c2, s in tests], fontsize=8.5)
    ax.set_ylim(len(tests) - .4, -.75)
    ax.set_xlim(.25, 1.02)
    ax.axvline(.5, ls='--', color='grey', lw=.5)
    n_perm = R.perm(*tests[0])['n_permutations'] if tests else 0
    ax.set_xlabel(f'AUROC ({n_perm} within-donor\nlabel permutations)')
    ax.legend([Line2D([], [], marker='o', ls='', color='#AAAAAA', ms=3), Line2D([], [], color='k', lw=1.2),
               Line2D([], [], marker='D', ls='', color='#C0392B', ms=5)],
              ['Permuted (null)', 'Null 95th percentile (P = 0.05)', f'Observed (all P ≤ {max(R.perm(*t)["p_auroc"] for t in tests):.2g})'],
              frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=1)


def fig_biology(R, out, comp='immune'):
    units = R.units(comp)
    colors = cohort_colors(units)
    meta = R.analysis(comp, 'paired_meta_analysis.tsv')
    per = R.analysis(comp, 'paired_differences_per_cohort.tsv')
    genes = meta[meta.type == 'gene'].copy()
    genes['p'] = genes.p.clip(lower=1e-300)
    fig = plt.figure(figsize=(215 * MM, 225 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.6)
    ax = fig.add_subplot(gs[0, 0])
    sig = (genes.fdr < .05) & (genes.cohorts_up.eq(genes.cohorts) | genes.cohorts_down.eq(genes.cohorts))
    ax.scatter(genes.log2_fold_change, -np.log10(genes.p), s=2, color='#BBBBBB', lw=0, rasterized=True)
    ax.scatter(genes.log2_fold_change[sig], -np.log10(genes.p[sig]), s=3, c=np.where(genes.log2_fold_change[sig] > 0, '#C0392B', '#2E86C1'), lw=0)
    lab = genes[sig].assign(score=lambda d: -np.log10(d.p)).sort_values('score', ascending=False)
    lab = pd.concat([lab[lab.log2_fold_change > 0].head(5), lab[lab.log2_fold_change < 0].head(5)])
    ax.set(xlabel='Pooled paired log2 fold change (tumour/normal)', ylabel='−log10 P (random effects)')
    ymax = (-np.log10(genes.p)).max() * 1.08
    ax.set_ylim(-ymax * .03, ymax)
    gap = ymax * .068
    for up, d in ((True, lab[lab.log2_fold_change > 0]), (False, lab[lab.log2_fold_change < 0])):
        d = d.sort_values('score', ascending=False)
        ys = list(-np.log10(d.p))
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - gap)
        x = d.log2_fold_change.max() + 0.9 if up else d.log2_fold_change.min() - 0.9
        for (_, r), y in zip(d.iterrows(), ys):
            ax.annotate(r.feature.split('::')[1], (r.log2_fold_change, -np.log10(r.p)), xytext=(x, y), fontsize=8, va='center',
                        ha='left' if up else 'right', arrowprops=dict(arrowstyle='-', lw=.4, color='#555555', shrinkA=1, shrinkB=1.5))
    lo, hi = ax.get_xlim()
    ax.set_xlim(min(lo, lab.log2_fold_change.min() - 2.4), hi)
    ax.text(0.99, 0.99, f'{int(sig.sum())} genes FDR<0.05 and\nconcordant in all cohorts', transform=ax.transAxes, va='top', ha='right', fontsize=8.5)
    panel(ax, 'a')
    ax = fig.add_subplot(gs[0, 1])
    top = genes[sig].sort_values('p').head(25).feature.tolist() or genes.sort_values('p').head(25).feature.tolist()
    hm = per[per.feature.isin(top)].pivot_table(index='feature', columns='cohort', values='mean_paired_difference').reindex(top) / np.log(2)
    lim = np.nanmax(np.abs(hm.to_numpy())) if hm.size else 1
    im = ax.imshow(hm.to_numpy(), cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_yticks(range(len(top)), [t.split('::')[1] for t in top], fontsize=8)
    ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=45, ha='right')
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, label='Paired log2 FC')
    ax.spines[:].set_visible(False)
    panel(ax, 'b')
    ax = fig.add_subplot(gs[1, 0])
    coef = pd.read_csv(R.root / 'ml' / comp / 'final_model_coefficients.tsv', sep='\t')
    stab = R.analysis(comp, 'selection_stability_loco.tsv').set_index('feature').fraction_of_folds
    c = coef.head(15).iloc[::-1]
    ax.barh([f.split('::')[1] + (' (prop)' if f.startswith('prop::') else '') for f in c.feature], c.coefficient,
            color=np.where(c.coefficient > 0, '#C0392B', '#2E86C1'))
    for i, f in enumerate(c.feature):
        ax.text(1.01, i, f'{stab.get(f, 0):.0%}', va='center', fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvline(0, color='k', lw=.5)
    ax.set_xlabel('Standardized coefficient (final model)')
    ax.set_title('Right: % of LOCO folds selecting feature', fontsize=9)
    panel(ax, 'c')
    ax = fig.add_subplot(gs[1, 1])
    oof = R.oof(comp, 'loco')
    p = oof[oof.model == 'logistic_regression'].pivot_table(index=['donor', 'cohort'], columns='truth', values='score').reset_index()
    for _, r in p.iterrows():
        ax.plot([0, 1], [r[0], r[1]], color=colors[r.cohort], lw=.6, alpha=.7)
    ax.scatter(np.zeros(len(p)), p[0], s=5, color=TISSUE['Normal'], zorder=3)
    ax.scatter(np.ones(len(p)), p[1], s=5, color=TISSUE['Tumor'], zorder=3)
    ax.set_xticks([0, 1], ['Adjacent normal', 'Tumour'])
    ax.set_xlim(-.3, 1.3)
    ax.set_ylabel('Out-of-cohort tumour probability')
    conc = float((p[1] > p[0]).mean())
    ax.set_title(f'Paired concordance {conc:.2f} ({int((p[1] > p[0]).sum())}/{len(p)} donors)')
    panel(ax, 'd')
    save(fig, out / 'Figure4_signature_biology')


def fig_supplement(R, out):
    comps = R.cfg['features']['compartments']
    fig, axes = plt.subplots(2, 2, figsize=(215 * MM, 185 * MM))
    plt.subplots_adjust(hspace=.5, wspace=.62)
    ax = axes[0, 0]
    vals = []
    for comp in comps:
        cp = json.loads((R.root / 'analysis' / comp / 'cohort_predictability.json').read_text())
        for k, v in cp.items():
            vals.append((f'{comp}: {k.replace("_", " ")}', v['balanced_accuracy'], v['chance']))
    ax.barh([v[0] for v in vals], [v[1] for v in vals], color='#7F8C8D')
    for i, v in enumerate(vals):
        ax.plot([v[2], v[2]], [i - .4, i + .4], color='k', ls='--', lw=.6)
    ax.set_xlabel('Cohort-of-origin balanced accuracy\n(dashed line: chance)')
    ax.set_xlim(0, 1)
    panel(ax, 'a')
    ax = axes[0, 1]
    permutation_panel(R, ax)
    panel(ax, 'b')
    ax = axes[1, 0]
    a, b = R.metrics('immune', 'loco').set_index('model'), R.metrics('immune', 'pooled').set_index('model')
    j = a.join(b, lsuffix='_loco', rsuffix='_pooled', how='inner')
    j = j[j.complete_oof_loco.astype(bool) & j.complete_oof_pooled.astype(bool)]
    ax.scatter(j.balanced_accuracy_pooled, j.balanced_accuracy_loco, s=8, color='#2C3E50')
    offsets = {'logistic_regression': ((-30, -48), 'right', 'logistic (prespecified)'), 'nested_algorithm_selection': ((-12, 14), 'right', 'nested selection'),
               'dummy_prior': ((10, 16), 'left', 'dummy prior'), 'svm_poly': ((6, -16), 'left', 'svm poly')}
    for name, (xy, ha, label) in offsets.items():
        if name in j.index:
            ax.annotate(label, (j.loc[name, 'balanced_accuracy_pooled'], j.loc[name, 'balanced_accuracy_loco']), fontsize=8.5, xytext=xy, ha=ha,
                        textcoords='offset points', arrowprops=dict(arrowstyle='-', lw=0.6, color='grey'))
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=.5)
    ax.set(xlabel='Pooled donor-CV balanced accuracy', ylabel='LOCO balanced accuracy', xlim=(0.3, 1.02), ylim=(0.3, 1.02))
    panel(ax, 'c')
    ax = axes[1, 1]
    oof = R.oof('immune', 'loco')
    p = oof[oof.model == 'logistic_regression']
    cm = confusion_matrix(p.truth, p.prediction, labels=[0, 1])
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for k in range(2):
            ax.text(k, i, f'{cm[i, k]}\n({cm[i, k] / cm[i].sum():.0%})', ha='center', va='center', color='white' if cm[i, k] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Normal', 'Tumour'])
    ax.set_yticks([0, 1], ['Normal', 'Tumour'])
    ax.set(xlabel='Predicted', ylabel='True', title='Logistic, LOCO (immune)')
    panel(ax, 'd')
    save(fig, out / 'FigureS1_batch_permutation_confusion')


def fig_subgroups(R, out, comp='immune'):
    sg = R.analysis(comp, 'subgroups_loco.tsv')
    sg = sg[sg.model == 'logistic_regression']
    fig, ax = plt.subplots(figsize=(150 * MM, 115 * MM))
    labels = [f'{r.variable}: {r.level} (n={r.n_donors})' for r in sg.itertuples()]
    ax.barh(labels, sg.balanced_accuracy, color='#7F8C8D', label='Balanced accuracy')
    if 'auroc' in sg:
        ax.scatter(sg.auroc, range(len(sg)), color='#C0392B', s=10, zorder=3, label='AUROC')
    ax.axvline(.5, ls='--', color='grey', lw=.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xlabel('Leave-one-cohort-out performance (prespecified logistic)')
    save(fig, out / 'FigureS2_subgroups')


def run(cfg):
    style()
    R = Results(cfg)
    out = R.root / 'publication'
    figs, tabs = out / 'figures', out / 'tables'
    figs.mkdir(parents=True, exist_ok=True)
    tables = write_tables(R, tabs)
    graphical.draw(R, figs / 'Figure1_graphical_abstract')
    fig_data(R, figs)
    fig_performance(R, figs)
    fig_biology(R, figs)
    fig_supplement(R, figs)
    fig_subgroups(R, figs)
    summary = numbers(R)
    (out / 'key_numbers.json').write_text(json.dumps(summary, indent=2, default=float))
    return tables, summary


def numbers(R):
    """Every number quoted in the manuscript is read from here."""
    s = {}
    libs = pd.read_csv(R.data / 'prepared' / 'libraries.tsv', sep='\t')
    qc = pd.read_csv(R.root / 'seurat' / 'qc_audit.tsv', sep='\t')
    s['libraries_eligible'] = len(libs)
    s['libraries_qc_pass'] = int((qc.status == 'complete').sum())
    s['cells_qc_pass'] = int(qc.retained_cells.fillna(0).sum())
    s['immune_cells_qc_pass'] = int(qc.immune_cells.fillna(0).sum())
    for comp in R.cfg['features']['compartments']:
        u = R.units(comp)
        sch = json.loads((R.root / 'features' / comp / 'schema.json').read_text())
        s[comp] = {'donors': int(u.donor_id.nunique()), 'units': len(u), 'cohorts': sch['cohorts'], 'shared_genes': sch['shared_genes'],
                   'histology': u.drop_duplicates('donor_id').histology.value_counts().to_dict(),
                   'enrichment': u.drop_duplicates('donor_id').enrichment.value_counts().to_dict()}
        for scheme in R.schemes(comp):
            m = R.metrics(comp, scheme).set_index('model')
            keep = [c for c in ['balanced_accuracy', 'balanced_accuracy_ci_low', 'balanced_accuracy_ci_high', 'auroc', 'auroc_ci_low', 'auroc_ci_high',
                                'auroc_mean_over_folds', 'paired_concordance', 'sensitivity_tumor', 'specificity_tumor', 'mcc', 'brier', 'n_donors'] if c in m]
            entry = {model: m.loc[model, keep].dropna().to_dict() for model in ('logistic_regression', 'nested_algorithm_selection', 'dummy_prior') if model in m.index}
            ok = m[m.complete_oof.astype(bool) & ~m.index.isin(['nested_algorithm_selection', 'dummy_prior'])]
            entry['models_complete'] = int(len(ok))
            entry['best_single_model_outer'] = {'model': ok.balanced_accuracy.idxmax(), 'balanced_accuracy': float(ok.balanced_accuracy.max())} if len(ok) else None
            entry['median_balanced_accuracy_all_models'] = float(ok.balanced_accuracy.median()) if len(ok) else None
            winners = json.loads((R.ml_dir(comp, scheme) / 'nested_selection.json').read_text())
            entry['nested_winners'] = [w['model'] for w in winners]
            perm = R.perm(comp, scheme)
            if perm:
                entry['permutation'] = perm
            s[comp][scheme] = entry
        meta = R.analysis(comp, 'paired_meta_analysis.tsv')
        g = meta[meta.type == 'gene']
        conc = g.cohorts_up.eq(g.cohorts) | g.cohorts_down.eq(g.cohorts)
        s[comp]['meta_genes_tested'] = int(len(g))
        s[comp]['meta_genes_fdr05'] = int((g.fdr < .05).sum())
        s[comp]['meta_genes_fdr05_concordant'] = int(((g.fdr < .05) & conc).sum())
        s[comp]['meta_proportions'] = meta[meta.type == 'proportion'][['feature', 'pooled_difference', 'se', 'p', 'fdr', 'I2', 'cohorts_up', 'cohorts_down', 'cohorts']].to_dict('records')
        s[comp]['cohort_predictability'] = json.loads((R.root / 'analysis' / comp / 'cohort_predictability.json').read_text())
        s[comp]['final_model_top_features'] = pd.read_csv(R.root / 'ml' / comp / 'final_model_coefficients.tsv', sep='\t').head(20).to_dict('records')
    return s
