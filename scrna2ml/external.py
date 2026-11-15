"""External validation: the final models, unchanged, applied to independent paired cohorts.

Each cohort has a sample sheet (config/external/<cohort>.tsv) listing one row per library: donor, tissue and the
deposited count triplet (URLs, or members of a series archive). Libraries are reduced to gene-expression rows,
run through the same Seurat workflow as the training libraries and scored by results/ml/<compartment>/final_model.joblib
without refitting, recalibration or a new threshold. Panel symbols renamed in newer annotations are recovered by
Ensembl ID from a training deposit; a specimen whose assay lacks any panel gene abstains.
"""
import gzip
import hashlib
import json
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse
from scipy.stats import binomtest

from . import download, ml
from .adapters import write_10x
from .config import REPO
from .features import COMPARTMENTS
from .seurat import R_SCRIPT

FILES = ('matrix', 'features', 'barcodes')


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _dirs(cfg):
    return Path(cfg['study']['data_dir']) / 'external', Path(cfg['study']['results_dir']) / 'external'


def sample_sheet(cfg, cohort):
    sheet = pd.read_csv(_resolve(cfg['external']['cohorts'][cohort]['samples']), sep='\t', dtype=str)
    if sheet.sample_id.duplicated().any() or sheet.groupby(['donor', 'tissue']).size().gt(1).any():
        raise ValueError(f'{cohort}: duplicated sample or donor x tissue in sample sheet')
    if not set(sheet.tissue) <= {'Tumor', 'Normal'}:
        raise ValueError(f'{cohort}: tissue must be Tumor or Normal')
    sheet['cohort'] = cohort
    sheet['library_id'] = cohort + '__' + sheet.sample_id
    sheet['donor_id'] = cohort + '__' + sheet.donor
    return sheet


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def fetch(cfg, workers=6):
    data, _ = _dirs(cfg)
    jobs = []
    for cohort, spec in cfg['external']['cohorts'].items():
        target = data / 'raw' / cohort
        target.mkdir(parents=True, exist_ok=True)
        sheet = sample_sheet(cfg, cohort)
        urls = [spec['archive']] if spec.get('archive') else []
        urls += [u for col in FILES for u in sheet[col] if not u.startswith('archive::')]
        jobs += [(cohort, u, target) for u in urls]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda j: download._fetch(*j), jobs))
    print(f'[external] {len(jobs)} files present', flush=True)


def _source(cfg, cohort, ref):
    """Local path of a sample-sheet file reference, extracting archive members on first use."""
    raw = _dirs(cfg)[0] / 'raw' / cohort
    if not ref.startswith('archive::'):
        return raw / ref.rsplit('/', 1)[1]
    member = ref.split('::', 1)[1]
    out = raw / 'extracted' / member
    if not out.exists():
        archive = raw / cfg['external']['cohorts'][cohort]['archive'].rsplit('/', 1)[1]
        out.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar, open(out.with_suffix('.part'), 'wb') as dst:
            dst.write(tar.extractfile(member).read())
        out.with_suffix('.part').replace(out)
    return out


def _reference_ids(cfg, panel):
    """Panel symbol -> Ensembl ID from a training deposit that lists both."""
    raw = Path(cfg['study']['data_dir']) / 'raw' / cfg['external']['symbol_reference']
    path = sorted(raw.glob('*features.tsv.gz'))[0]
    ref = pd.read_csv(path, sep='\t', header=None, dtype=str)
    ref = ref[ref[2] == 'Gene Expression'] if ref.shape[1] > 2 else ref
    ref['symbol'] = ref[1].str.replace('_', '-')
    ids = ref[ref.symbol.isin(panel)].groupby('symbol')[0].agg(set)
    return {s: next(iter(v)) for s, v in ids.items() if len(v) == 1}


def _rename(features, panel, ref_ids):
    """Rename source symbols of panel genes that a newer annotation renamed, matched by unique Ensembl ID."""
    symbols = features[1].str.replace('_', '-')
    present = set(symbols)
    mapping = {}
    for target in sorted(panel - present):
        gid = ref_ids.get(target)
        rows = np.flatnonzero(features[0].to_numpy() == gid) if gid else []
        if len(rows) != 1:
            continue
        old = symbols.iloc[rows[0]]
        if old in panel or (symbols == old).sum() != 1:
            continue
        mapping[old] = (target, gid)
    return [mapping[s][0] if s in mapping else s for s in features[1]], mapping


def _write_library(library_id, src, keep, genes, out):
    with gzip.open(src['matrix'], 'rb') as f:
        mat = sparse.csr_matrix(spio.mmread(f))
    with gzip.open(src['barcodes'], 'rt') as f:
        barcodes = [line.strip() for line in f]
    if mat.shape != (len(keep), len(barcodes)):
        raise ValueError(f'{library_id}: matrix {mat.shape} vs {len(keep)} features, {len(barcodes)} barcodes')
    write_10x(out, mat[keep], genes, barcodes)


def prepare(cfg, panel):
    data, res = _dirs(cfg)
    ref_ids = _reference_ids(cfg, panel)
    records, renames, jobs = [], [], []
    for cohort in cfg['external']['cohorts']:
        for r in sample_sheet(cfg, cohort).itertuples():
            out = data / 'prepared' / cohort / r.sample_id
            src = {col: _source(cfg, cohort, getattr(r, col)) for col in FILES}
            features = pd.read_csv(src['features'], sep='\t', header=None, dtype=str)
            if features.shape[1] < 2:
                raise ValueError(f'{r.library_id}: feature table without gene IDs')
            keep = (features[2] == 'Gene Expression').to_numpy() if features.shape[1] > 2 else np.ones(len(features), bool)
            features = features[keep].reset_index(drop=True)
            genes, mapping = _rename(features, panel, ref_ids)
            renames += [{'cohort': cohort, 'library_id': r.library_id, 'ensembl_id': gid, 'source_symbol': old,
                         'panel_symbol': new} for old, (new, gid) in sorted(mapping.items())]
            if not (out / 'complete.json').exists():
                jobs.append((r.library_id, src, keep, genes, out))
            records.append({'library_id': r.library_id, 'cohort': cohort, 'donor_id': r.donor_id, 'tissue': r.tissue,
                            'histology': r.histology, 'chemistry': r.chemistry, 'path': str(out),
                            'sources': '|'.join(str(src[c]) for c in FILES)})
    with ProcessPoolExecutor(max_workers=int(cfg['qc'].get('workers', 4)) * 2) as pool:
        for fut in [pool.submit(_write_library, *job) for job in jobs]:
            fut.result()
    libs = pd.DataFrame(records)
    res.mkdir(parents=True, exist_ok=True)
    libs.to_csv(res / 'libraries.tsv', sep='\t', index=False)
    pd.DataFrame(renames, columns=['cohort', 'library_id', 'ensembl_id', 'source_symbol', 'panel_symbol']).to_csv(
        res / 'symbol_mapping.tsv', sep='\t', index=False)
    print(f'[external] prepared {len(libs)} libraries, {len(renames)} renamed panel symbols', flush=True)
    return libs


def run_seurat(cfg, libs):
    out = _dirs(cfg)[1] / 'seurat'
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame({'sample_id': libs.library_id, **{c: [str(Path(p) / f'{c}.{s}') for p in libs.path]
                             for c, s in (('matrix', 'mtx.gz'), ('genes', 'tsv.gz'), ('barcodes', 'tsv.gz'))}})
    manifest.to_csv(out / 'manifest.tsv', sep='\t', index=False)
    (out / 'qc.json').write_text(json.dumps(cfg['qc'], indent=2))
    subprocess.run(['Rscript', str(R_SCRIPT), str(out / 'manifest.tsv'), str(out), str(out / 'qc.json')], check=True)


def _assayed(lib_paths):
    sets = [set(pd.read_csv(Path(p) / 'genes.tsv.gz', header=None)[0].astype(str).str.replace('_', '-')) for p in lib_paths]
    return set.intersection(*sets)


def score(cfg, libs, comp, bundle):
    """One row per donor x tissue: features built exactly as in features.build, then the frozen pipeline."""
    spec, seurat_dir = COMPARTMENTS[comp], _dirs(cfg)[1] / 'seurat'
    feats = bundle['features']
    panel = [f[6:] for f in feats if f.startswith('gene::')]
    props = [f[6:] for f in feats if f.startswith('prop::')]
    min_cells, threshold = int(cfg['features']['min_cells']), float(cfg['external']['threshold'])
    rows, X = [], []
    for donor, d in libs.groupby('donor_id', sort=True):
        per = {}
        for tissue, t in d.groupby('tissue'):
            info = [json.loads((seurat_dir / lib / 'complete.json').read_text()) for lib in t.library_id]
            if any(i['status'] != 'complete' for i in info):
                per[tissue] = None
                continue
            pb = [pd.read_csv(seurat_dir / lib / f'{lib}{spec["pseudobulk"]}', sep='\t', index_col='gene')['count']
                  .groupby(level=0).sum() for lib in t.library_id if (seurat_dir / lib / f'{lib}{spec["pseudobulk"]}').exists()]
            comp_counts = sum(pd.read_csv(seurat_dir / lib / f'{lib}{spec["composition"]}', sep='\t')
                              .set_index('celltype')['count'] for lib in t.library_id)
            per[tissue] = (pb, comp_counts, t, _assayed(t.path))
        for tissue, t in d.groupby('tissue'):
            base = {'cohort': t.cohort.iloc[0], 'donor_id': donor, 'tissue': tissue, 'compartment': comp,
                    'libraries': '|'.join(t.library_id), 'chemistry': t.chemistry.iloc[0], 'histology': t.histology.iloc[0]}
            if set(per) != {'Tumor', 'Normal'} or any(v is None for v in per.values()):
                rows.append({**base, 'status': 'abstain: no QC-passing tumour/normal pair'})
                continue
            pb, comp_counts, _, assayed = per[tissue]
            cells = {k: int(v[1].sum()) for k, v in per.items()}
            missing = sorted(set(panel) - assayed)
            extra = set(comp_counts[comp_counts > 0].index) - set(props)
            if extra:
                raise ValueError(f'{donor}: cell types {extra} absent from the training features')
            if min(cells.values()) < min_cells or not pb:
                rows.append({**base, 'cells': cells[tissue], 'status': f'abstain: fewer than {min_cells} {comp} cells in a tissue'})
                continue
            if missing:
                rows.append({**base, 'cells': cells[tissue], 'missing_panel_genes': len(missing), 'status': 'abstain: incomplete gene panel'})
                continue
            counts = pd.concat(pb, axis=1).fillna(0).sum(axis=1).reindex(panel, fill_value=0).astype(float)
            x = np.log1p(counts / counts.sum() * 1e6).tolist() + (comp_counts.reindex(props, fill_value=0) / comp_counts.sum()).tolist()
            X.append(x)
            rows.append({**base, 'cells': cells[tissue], 'missing_panel_genes': 0, 'panel_umi': int(counts.sum()), 'status': 'scored'})
    pred = pd.DataFrame(rows)
    scored = pred.status.eq('scored').to_numpy()
    if scored.any():
        p = bundle['pipeline'].predict_proba(pd.DataFrame(X, columns=feats))[:, 1]
        pred.loc[scored, 'score'] = p
        pred.loc[scored, 'prediction'] = np.where(p > threshold, 'Tumor', 'Normal')
        np.testing.assert_array_equal(bundle['pipeline'].predict(pd.DataFrame(X, columns=feats)) == 1, p > 0.5)
    return pred


def metrics(cfg, pred):
    n_boot, seed = cfg['ml']['bootstrap'], cfg['study']['seed']
    rows = []
    groups = [(c, comp, d) for (c, comp), d in pred.groupby(['cohort', 'compartment'])]
    groups += [('All external', comp, d) for comp, d in pred.groupby('compartment')]
    for cohort, comp, d in groups:
        s = d[d.status == 'scored']
        s = s[s.groupby('donor_id').tissue.transform('nunique') == 2]
        row = {'cohort': cohort, 'compartment': comp, 'donors_listed': d.donor_id.nunique(), 'donors_scored': s.donor_id.nunique()}
        if s.empty:
            rows.append(row)
            continue
        frame = pd.DataFrame({'donor': s.donor_id, 'truth': s.tissue.eq('Tumor').astype(int),
                              'prediction': s.prediction.eq('Tumor').astype(int), 'score': s.score.astype(float),
                              'fold': s.cohort, 'probability': True})
        row.update(ml.summarize(frame, n_boot, seed))
        wide = frame.pivot(index='donor', columns='truth', values='score')
        higher = int((wide[1] > wide[0]).sum())
        row.update(pairs_tumour_higher=higher, sign_test_p=binomtest(higher, len(wide), 0.5, alternative='greater').pvalue,
                   tn=int(((frame.truth == 0) & (frame.prediction == 0)).sum()), fp=int(((frame.truth == 0) & (frame.prediction == 1)).sum()),
                   fn=int(((frame.truth == 1) & (frame.prediction == 0)).sum()), tp=int(((frame.truth == 1) & (frame.prediction == 1)).sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def run(cfg):
    if not cfg.get('external'):
        return
    _, res = _dirs(cfg)
    bundles, digests = {}, {}
    for comp in cfg['features']['compartments']:
        path = Path(cfg['study']['results_dir']) / 'ml' / comp / 'final_model.joblib'
        digests[comp] = _sha256(path)
        bundles[comp] = joblib.load(path)
    panels = [{f[6:] for f in b['features'] if f.startswith('gene::')} for b in bundles.values()]
    if any(p != panels[0] for p in panels):
        raise ValueError('Compartment models use different gene panels')
    if set(cfg['external']['cohorts']) & set(cfg['datasets']):
        raise ValueError('An external cohort is also a training cohort')
    fetch(cfg)
    libs = prepare(cfg, panels[0])
    run_seurat(cfg, libs)
    pred = pd.concat([score(cfg, libs, comp, bundles[comp]) for comp in bundles], ignore_index=True)
    pred.to_csv(res / 'predictions.tsv', sep='\t', index=False)
    table = metrics(cfg, pred)
    table.to_csv(res / 'metrics.tsv', sep='\t', index=False)
    after = {comp: _sha256(Path(cfg['study']['results_dir']) / 'ml' / comp / 'final_model.joblib') for comp in bundles}
    if after != digests:
        raise RuntimeError('A final model changed during external scoring')
    (res / 'checks.json').write_text(json.dumps({
        'final_model_sha256': digests, 'threshold': cfg['external']['threshold'], 'panel_genes': len(panels[0]),
        'libraries': len(libs), 'scored_rows': int(pred.status.eq('scored').sum()),
        'abstentions': pred.loc[pred.status != 'scored', 'status'].value_counts().to_dict(),
        'training_cohorts_disjoint': True}, indent=2))
    print(table[['cohort', 'compartment', 'donors_scored', 'balanced_accuracy', 'auroc', 'pairs_tumour_higher']].to_string(index=False), flush=True)
