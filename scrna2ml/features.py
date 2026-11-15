"""Donor x tissue feature matrices from per-library Seurat exports.

Unit of analysis: one tumour and one adjacent-normal specimen per donor (paired design).
Within a donor both tissues must come from the same cell fraction (CD45+-enriched preferred for the
immune compartment, otherwise unsorted); libraries of that fraction are summed. Features are
log1p CPM over genes listed by every input library plus cell-type proportions. No labels are used here
except to assert pairing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

COMPARTMENTS = {
    'immune': {'fractions': ['CD45+', 'unsorted'], 'pseudobulk': '_immune_pseudobulk.tsv', 'composition': '_immune_composition.tsv'},
    'whole': {'fractions': ['unsorted'], 'pseudobulk': '_pseudobulk.tsv', 'composition': '_composition.tsv'},
}


def _files(seurat_dir, library_id, suffix):
    return Path(seurat_dir) / library_id / f'{library_id}{suffix}'


def _read_counts(path):
    s = pd.read_csv(path, sep='\t', index_col='gene')['count']
    return s.groupby(level=0).sum()


def build(cfg, compartment):
    spec = COMPARTMENTS[compartment]
    results = Path(cfg['study']['results_dir'])
    seurat_dir = results / 'seurat'
    min_cells = int(cfg['features']['min_cells'])
    qc = pd.read_csv(seurat_dir / 'qc_audit.tsv', sep='\t')
    qc = qc[(qc.status == 'complete') & qc.enrichment.isin(spec['fractions'])]
    units, counts, comps, log = [], [], [], []
    for donor, d in qc.groupby('donor_id', sort=True):
        chosen = None
        for fraction in spec['fractions']:
            sub = d[d.enrichment == fraction]
            if set(sub.tissue) == {'Tumor', 'Normal'}:
                chosen = (fraction, sub)
                break
        if chosen is None:
            log.append({'donor_id': donor, 'compartment': compartment, 'reason': 'No tumour/normal pair from the same cell fraction after QC'})
            continue
        fraction, sub = chosen
        per_tissue = {}
        for tissue, t in sub.groupby('tissue'):
            pb = [_read_counts(_files(seurat_dir, lib, spec['pseudobulk'])) for lib in t.library_id
                  if _files(seurat_dir, lib, spec['pseudobulk']).exists()]
            comp = sum(pd.read_csv(_files(seurat_dir, lib, spec['composition']), sep='\t').set_index('celltype')['count'] for lib in t.library_id)
            per_tissue[tissue] = (pb, comp, t)
        cell_counts = {tissue: int(v[1].sum()) for tissue, v in per_tissue.items()}
        if min(cell_counts.values()) < min_cells or any(len(v[0]) == 0 for v in per_tissue.values()):
            log.append({'donor_id': donor, 'compartment': compartment, 'reason': f'Fewer than {min_cells} {compartment} cells in a tissue ({cell_counts})'})
            continue
        for tissue in ('Tumor', 'Normal'):
            pb, comp, t = per_tissue[tissue]
            counts.append(pd.concat(pb, axis=1).fillna(0).sum(axis=1))
            comps.append(comp)
            first = t.iloc[0]
            units.append({'unit_id': f'{donor}__{tissue}', 'donor_id': donor, 'cohort': first.cohort, 'tissue': tissue,
                          'histology': first.histology, 'enrichment': fraction, 'n_libraries': len(t),
                          'libraries': '|'.join(t.library_id), 'cells': cell_counts[tissue]})
    units = pd.DataFrame(units)
    if units.empty:
        raise ValueError(f'No paired units for compartment {compartment}')
    # Genes absent from a deposited input matrix (author gene filtering, e.g. GSE117570) are unmeasured, not zero,
    # so only genes listed in the input of every library used are kept. The panel uses gene lists only (no expression
    # values, no labels); analysis.fold_local_panel re-derives it from training cohorts alone. Seurat renames '_' to '-'.
    used = set('|'.join(units.libraries).split('|'))
    measured = None
    for path in qc.loc[qc.library_id.isin(used), 'path']:
        g = set(pd.read_csv(Path(path) / 'genes.tsv.gz', header=None)[0].astype(str).str.replace('_', '-'))
        measured = g if measured is None else measured & g
    shared = sorted(measured)
    mat = pd.DataFrame([c.reindex(shared, fill_value=0).to_numpy() for c in counts], columns=shared)
    logcpm = np.log1p(mat.div(mat.sum(axis=1), axis=0) * 1e6)
    logcpm.columns = ['gene::' + g for g in shared]
    comp = pd.DataFrame(comps).fillna(0)
    prop = comp.div(comp.sum(axis=1), axis=0)
    prop.columns = ['prop::' + c for c in prop.columns]
    X = pd.concat([logcpm.reset_index(drop=True), prop.reset_index(drop=True)], axis=1)
    units['label'] = units.tissue
    assert units.groupby('donor_id').tissue.apply(lambda s: sorted(s) == ['Normal', 'Tumor']).all()
    out = results / 'features' / compartment
    out.mkdir(parents=True, exist_ok=True)
    units.to_csv(out / 'units.tsv', sep='\t', index=False)
    X.to_csv(out / 'X.tsv.gz', sep='\t', index=False)
    mat.to_csv(out / 'counts.tsv.gz', sep='\t', index=False)
    pd.DataFrame(log, columns=['donor_id', 'compartment', 'reason']).to_csv(out / 'unit_exclusions.tsv', sep='\t', index=False)
    (out / 'schema.json').write_text(json.dumps({
        'unit': 'donor x tissue; tumour and adjacent normal from the same donor and cell fraction',
        'expression': 'log1p(1e6 * summed raw pseudobulk count / sum over genes listed in the input of every library used)',
        'measured_in_all_input_libraries': len(measured),
        'proportions': f'fixed provisional marker labels within the {compartment} compartment',
        'shared_genes': len(shared), 'units': len(units), 'donors': int(units.donor_id.nunique()),
        'cohorts': units.groupby('cohort').donor_id.nunique().to_dict(), 'min_cells_per_unit': min_cells}, indent=2))
    print(f'[features] {compartment}: {units.donor_id.nunique()} donors, {len(shared)} shared genes, {len(log)} donors excluded', flush=True)
    return units, X
