"""Graphical abstract / study overview (Figure 1). All numbers are read from pipeline outputs."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import roc_curve

MM = 1 / 25.4
INK = '#1B2631'
GREY = '#3F4A54'
ACCENT = '#C0392B'
GREEN = '#4C9F70'
W, H, Y = 0.228, 0.80, 0.04
XS = [0.008, 0.256, 0.504, 0.752]
COLS = ['#34495E', '#16A085', '#8E44AD', ACCENT]
HEADER = 0.085


def _box(ax, x, title, color):
    ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle='round,pad=0,rounding_size=0.015', fc='white', ec=color, lw=1.8))
    ax.add_patch(FancyBboxPatch((x, Y + H - HEADER), W, HEADER, boxstyle='round,pad=0,rounding_size=0.015', fc=color, ec=color, lw=1.8))
    ax.add_patch(Rectangle((x, Y + H - HEADER), W, HEADER / 2, fc=color, ec=color, lw=0))
    ax.text(x + W / 2, Y + H - HEADER / 2, title, ha='center', va='center', color='white', fontsize=14, fontweight='bold')


def _lines(ax, x, y, lines, size=11, color=GREY, weight='normal', gap=0.052):
    for i, line in enumerate(lines):
        ax.text(x + W / 2, y - i * gap, line, ha='center', va='center', fontsize=size, color=color, fontweight=weight)


def draw(R, path):
    cfg = R.cfg
    units = R.units('immune')
    qc = pd.read_csv(R.root / 'seurat' / 'qc_audit.tsv', sep='\t')
    m = R.metrics('immune', 'loco').set_index('model').loc['logistic_regression']
    oof = R.oof('immune', 'loco')
    p = oof[oof.model == 'logistic_regression']
    cohorts = units.drop_duplicates('donor_id').groupby('cohort').size().sort_values(ascending=False)
    n_cells = int(qc.retained_cells.fillna(0).sum())
    n_libs = int((qc.status == 'complete').sum())
    sch = json.loads((R.root / 'features' / 'immune' / 'schema.json').read_text())
    from .models import registry
    n_models = len([m for m in registry()[0] if m != 'dummy_prior'])
    q = cfg['qc']

    fig = plt.figure(figsize=(250 * MM, 140 * MM))
    ax = fig.add_axes([0, 0, 1, 0.9])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.text(0.5, 0.86, 'Immune-compartment classifiers of NSCLC tumour vs matched adjacent lung,\nvalidated across independent scRNA-seq cohorts',
             ha='center', va='center', fontsize=16, fontweight='bold', color=INK, linespacing=1.25)
    titles = ['1  Public scRNA-seq', '2  Identical QC', '3  Donor features', '4  Cohort-held-out ML']
    for x, t, c in zip(XS, titles, COLS):
        _box(ax, x, t, c)
    for a, b in zip(XS[:-1], XS[1:]):
        ax.add_patch(FancyArrowPatch((a + W + 0.002, Y + H / 2), (b - 0.002, Y + H / 2), arrowstyle='-|>', mutation_scale=16, color=INK, lw=1.6))
    top = Y + H - HEADER

    # 1 cohorts
    x0 = XS[0]
    _lines(ax, x0, top - 0.06, [f'{len(cohorts)} cohorts', f'{units.donor_id.nunique()} paired donors'], size=12.5, color=INK, weight='bold')
    maxn = cohorts.max()
    for i, (c, n) in enumerate(cohorts.items()):
        yy = top - 0.175 - i * 0.05
        ax.text(x0 + 0.012, yy, c, fontsize=10.5, va='center', color=INK)
        ax.add_patch(Rectangle((x0 + 0.112, yy - 0.019), 0.085 * n / maxn, 0.038, color=COLS[0], alpha=.8))
        ax.text(x0 + 0.117 + 0.085 * n / maxn, yy, str(n), fontsize=10.5, va='center', color=INK)
    for k, (lab, col) in enumerate([('Tumour', ACCENT), ('Adjacent normal', GREEN)]):
        yb = Y + 0.205 - k * 0.042
        ax.add_patch(Circle((x0 + 0.06, yb), 0.009, color=col))
        ax.text(x0 + 0.075, yb, lab, fontsize=10.5, va='center', color=INK)
    _lines(ax, x0, Y + 0.085, ['Same donor, same fraction', '(CD45+ or non-CD45-selected)'], size=10.5, gap=0.042)

    # 2 QC + annotation
    x0 = XS[1]
    rng = np.random.default_rng(3)
    centers = {'T': (0.05, 0.68), 'NK': (0.115, 0.725), 'B': (0.18, 0.68), 'Myeloid': (0.165, 0.56), 'Other': (0.055, 0.56)}
    pal = {'T': '#2E86C1', 'NK': '#1ABC9C', 'B': '#F39C12', 'Myeloid': '#C0392B', 'Other': '#95A5A6'}
    for k, (cx, cy) in centers.items():
        pts = rng.normal(0, 0.012, (45, 2))
        ax.scatter(x0 + cx + pts[:, 0], Y + cy * H - 0.05 + pts[:, 1] * 1.5, s=5, color=pal[k], lw=0)
        dy = -0.05 if k in ('Other', 'Myeloid') else 0.05
        ax.text(x0 + cx, Y + cy * H - 0.05 + dy, k, fontsize=10, ha='center', va='center', color=pal[k], fontweight='bold')
    _lines(ax, x0, Y + 0.30, [f'{n_libs} libraries', f'{n_cells:,} cells'], size=12.5, color=INK, weight='bold')
    _lines(ax, x0, Y + 0.17, [f"UMI ≥{q['min_counts']}, genes ≥{q['min_features']}", f"mitochondrial ≤{q['max_percent_mt']}%",
                              'label-blind marker scores'], size=10.5, gap=0.045)

    # 3 features
    x0 = XS[2]
    hm = rng.normal(0, 1, (9, 12))
    hm[:, ::2] -= 0.9
    hm[:, 1::2] += 0.9
    hax = fig.add_axes([x0 + 0.045, (Y + 0.44) * 0.9, 0.145, 0.25 * 0.9])
    hax.imshow(hm, cmap='RdBu_r', aspect='auto', vmin=-2.5, vmax=2.5)
    hax.set_xticks(range(0, 12, 2), ['N/T'] * 6, fontsize=8.5)
    hax.tick_params(length=0, pad=2)
    hax.set_yticks([])
    for s in hax.spines.values():
        s.set_visible(False)
    hax.set_ylabel('genes', fontsize=10, labelpad=3)
    _lines(ax, x0, Y + 0.30, ['Immune pseudobulk', '+ cell proportions'], size=12.5, color=INK, weight='bold')
    _lines(ax, x0, Y + 0.17, [f"log-CPM over {sch['shared_genes']:,} genes", 'listed by every assay', '(expression-blind panel)'], size=10.5, gap=0.045)

    # 4 validation
    x0 = XS[3]
    inset = fig.add_axes([x0 + 0.06, (Y + 0.43) * 0.9, 0.13, 0.26 * 0.9])
    fpr, tpr, _ = roc_curve(p.truth, p.score)
    inset.plot(fpr, tpr, color=ACCENT, lw=2)
    inset.plot([0, 1], [0, 1], ls='--', color='grey', lw=.8)
    inset.set_xticks([0, 1])
    inset.set_yticks([0, 1])
    inset.tick_params(labelsize=9, length=2, pad=1)
    inset.set_xlabel('1 − specificity', fontsize=9.5, labelpad=0)
    inset.set_ylabel('sensitivity', fontsize=9.5, labelpad=0)
    for s in ('top', 'right'):
        inset.spines[s].set_visible(False)
    _lines(ax, x0, Y + 0.30, [f'AUROC {m.auroc:.2f}', f'balanced acc. {m.balanced_accuracy:.2f}'], size=12.5, color=ACCENT, weight='bold')
    _lines(ax, x0, Y + 0.17, ['leave-one-cohort-out', f'{n_models} algorithms, nested', f'tumour > normal in {m.paired_concordance:.0%}'], size=10.5, gap=0.045)

    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{path}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
