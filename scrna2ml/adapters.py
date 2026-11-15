"""Source-specific adapters: raw public downloads -> standardized per-library 10x matrices.

Each adapter yields library records and writes data/prepared/<cohort>/<library>/{matrix.mtx.gz,genes.tsv.gz,barcodes.tsv.gz}.
Only gene-expression features are kept (antibody/hashtag rows removed). Original downloads are never modified.
Inclusion rules are applied here and every excluded library is logged with a reason.
"""
import gzip
import io
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse

PREFILTER_UMI = 100  # far below the Seurat QC floor (500 UMIs); only drops empty droplets in unfiltered matrices
HTO_MIN_COUNTS = 10
HTO_MIN_FRACTION = 0.8


def write_10x(outdir, mat, genes, barcodes):
    """mat: genes x cells sparse integer counts."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mat = sparse.csc_matrix(mat)
    if mat.shape != (len(genes), len(barcodes)):
        raise ValueError(f'Shape mismatch {mat.shape} vs {len(genes)} genes, {len(barcodes)} barcodes')
    data = mat.data
    if data.size and (np.any(data < 0) or not np.allclose(data, np.round(data))):
        raise ValueError(f'{outdir}: matrix is not raw non-negative integer counts')
    mat = mat.astype(np.int64)
    if len(set(barcodes)) != len(barcodes):
        raise ValueError(f'{outdir}: duplicated barcodes')
    with gzip.open(outdir / 'matrix.mtx.gz', 'wb') as f:
        spio.mmwrite(f, mat, field='integer')
    pd.Series(genes).to_csv(outdir / 'genes.tsv.gz', index=False, header=False, compression='gzip')
    pd.Series(barcodes).to_csv(outdir / 'barcodes.tsv.gz', index=False, header=False, compression='gzip')
    (outdir / 'complete.json').write_text(json.dumps({'genes': int(mat.shape[0]), 'cells': int(mat.shape[1]), 'nnz': int(mat.nnz)}))


def _done(outdir):
    return (Path(outdir) / 'complete.json').exists()


def _prefilter(mat, barcodes, min_umi=PREFILTER_UMI):
    totals = np.asarray(mat.sum(axis=0)).ravel()
    keep = totals >= min_umi
    return mat[:, keep], [b for b, k in zip(barcodes, keep) if k]


def record(cohort, library, donor, tissue, histology, enrichment, path, **extra):
    return dict(cohort=cohort, library_id=f'{cohort}__{library}', donor_id=f'{cohort}__{donor}', tissue=tissue,
                histology=histology, enrichment=enrichment, path=str(path), **extra)


# --------------------------------------------------------------------------- GSE117570
def gse117570(raw, out):
    histology = {'P1': 'LUAD', 'P2': 'LUSC', 'P3': 'LUAD', 'P4': 'LUAD'}
    recs = []
    with tarfile.open(raw / 'GSE117570_RAW.tar') as tar:
        for m in tar.getmembers():
            g = re.match(r'(GSM\d+)_(P\d)_(Tumor|Normal)_processed_data\.txt\.gz', m.name)
            if not g:
                continue
            gsm, donor, tissue = g.groups()
            dest = out / gsm
            if not _done(dest):
                df = pd.read_csv(io.BytesIO(gzip.decompress(tar.extractfile(m).read())), sep='\t', index_col=0)
                df = df.groupby(level=0).sum()
                write_10x(dest, sparse.csc_matrix(df.to_numpy()), df.index.astype(str).tolist(), df.columns.astype(str).tolist())
            recs.append(record('GSE117570', gsm, donor, tissue, histology[donor], 'unsorted', dest, chemistry="10x 3'", source_sample=gsm))
    return recs, []


# --------------------------------------------------------------------------- GSE131907
def gse131907(raw, out):
    ann = pd.read_csv(raw / 'GSE131907_Lung_Cancer_cell_annotation.txt.gz', sep='\t', usecols=['Index', 'Sample', 'Sample_Origin'])
    lung = ann[ann.Sample_Origin.isin(['tLung', 'nLung'])]
    ids = lung.Sample.str.extract(r'LUNG_([TN])(\d+)')
    lung = lung.assign(tissue=ids[0].map({'T': 'Tumor', 'N': 'Normal'}).to_numpy(), donor=('P' + ids[1]).to_numpy())
    pairs = lung.groupby('donor').tissue.agg(lambda s: set(s))
    paired = sorted(pairs[pairs.apply(lambda s: s == {'Tumor', 'Normal'})].index)
    excl = [dict(cohort='GSE131907', library=s, reason='Lung sample without matched tumour/normal partner') for s in sorted(lung.Sample.unique()) if ('P' + re.sub(r'\D', '', s)) not in paired]
    excl += [dict(cohort='GSE131907', library=s, reason=f'Not primary lung tissue ({o})') for s, o in ann[~ann.Sample_Origin.isin(['tLung', 'nLung'])].drop_duplicates('Sample')[['Sample', 'Sample_Origin']].itertuples(index=False)]
    keep = lung[lung.donor.isin(paired)]
    samples = sorted(keep.Sample.unique())
    if not all(_done(out / s) for s in samples):
        cells = out / '_cells.tsv'
        out.mkdir(parents=True, exist_ok=True)
        keep[['Index', 'Sample']].to_csv(cells, sep='\t', index=False)
        script = Path(__file__).parent / 'R' / 'split_gse131907.R'
        subprocess.run(['Rscript', str(script), str(raw / 'GSE131907_Lung_Cancer_raw_UMI_matrix.rds.gz'), str(cells), str(out)], check=True)
    recs = []
    for s in samples:
        meta = keep[keep.Sample == s].iloc[0]
        recs.append(record('GSE131907', s, meta.donor, meta.tissue, 'LUAD', 'unsorted', out / s, chemistry="10x 3' v2", source_sample=s))
    return recs, excl


# --------------------------------------------------------------------------- GSE123902
def gse123902(raw, out):
    paired = {'LX675', 'LX682', 'LX684'}  # untreated primary tumours with matched adjacent normal lung (GEO characteristics)
    recs, excl = [], []
    with tarfile.open(raw / 'GSE123902_RAW.tar') as tar:
        for m in tar.getmembers():
            g = re.match(r'(GSM\d+)_MSK_(LX\w+?)_(PRIMARY_TUMOUR|NORMAL|METASTASIS)_dense\.csv\.gz', m.name)
            if not g:
                continue
            gsm, donor, kind = g.groups()
            if kind == 'METASTASIS' or donor not in paired:
                excl.append(dict(cohort='GSE123902', library=gsm, reason='Metastasis' if kind == 'METASTASIS' else 'No matched normal (or post-chemotherapy)'))
                continue
            dest = out / gsm
            if not _done(dest):
                df = pd.read_csv(io.BytesIO(gzip.decompress(tar.extractfile(m).read())), index_col=0)
                write_10x(dest, sparse.csc_matrix(df.to_numpy().T), df.columns.astype(str).tolist(), df.index.astype(str).tolist())
            recs.append(record('GSE123902', gsm, donor, 'Tumor' if kind == 'PRIMARY_TUMOUR' else 'Normal', 'LUAD', 'unsorted', dest, chemistry="10x 3' v2", source_sample=gsm))
    return recs, excl


# --------------------------------------------------------------------------- GSE198099
def gse198099(raw, out):
    mapping = {'Ca1': ('patient1', 'Tumor'), 'N1': ('patient1', 'Normal'), 'Ca2': ('patient2', 'Tumor'), 'N2': ('patient2', 'Normal')}
    recs, excl = [], []
    with tarfile.open(raw / 'GSE198099_RAW.tar') as tar, tempfile.TemporaryDirectory() as tmp:
        tar.extractall(tmp)
        tmp = Path(tmp)
        for gsm_mtx in sorted(tmp.glob('*_matrix.mtx.gz')):
            gsm, lib = re.match(r'(GSM\d+)_(\w+?)_matrix', gsm_mtx.name).groups()
            if lib not in mapping:
                excl.append(dict(cohort='GSE198099', library=gsm, reason='Lymph node'))
                continue
            donor, tissue = mapping[lib]
            dest = out / gsm
            if not _done(dest):
                mat = spio.mmread(str(gsm_mtx)).tocsc()
                feat = pd.read_csv(tmp / f'{gsm}_{lib}_genes.tsv.gz', sep='\t', header=None)
                bars = pd.read_csv(tmp / f'{gsm}_{lib}_barcodes.tsv.gz', header=None)[0].astype(str).tolist()
                write_10x(dest, mat, feat[1 if feat.shape[1] > 1 else 0].astype(str).tolist(), bars)
            recs.append(record('GSE198099', gsm, donor, tissue, 'NSCLC', 'unsorted', dest, chemistry="10x 3'", source_sample=gsm))
    return recs, excl


# --------------------------------------------------------------------------- E-MTAB-13526
EMTAB_FRACTION = {'CD45+': 'CD45+', 'CD235a-': 'unsorted', 'MDSC': 'MDSC', 'CD45+ + MDSC': 'CD45+MDSC'}


def emtab13526(raw, out):
    sdrf = pd.read_csv(raw / 'E-MTAB-13526.sdrf.txt', sep='\t')
    sdrf['lib'] = sdrf['Derived Array Data File'].str.replace('-matrix.mtx.gz', '', regex=False)
    meta = sdrf.drop_duplicates('lib').set_index('lib')
    recs, excl = [], []
    hist = {'LUAD': 'LUAD', 'LUSC': 'LUSC', 'LC': 'NSCLC'}
    for lib, r in meta.iterrows():
        if not lib.startswith('P'):
            excl.append(dict(cohort='E-MTAB-13526', library=lib, reason='Deceased healthy donor (no matched tumour)'))
            continue
        fraction = EMTAB_FRACTION[r['Characteristics[FACS]']]
        if fraction in ('MDSC', 'CD45+MDSC'):
            excl.append(dict(cohort='E-MTAB-13526', library=lib, reason=f'Sorted on a myeloid subset ({fraction}); composition not comparable'))
            continue
        dest = out / lib
        tissue = 'Tumor' if r['Characteristics[sampling site]'] == 'tumor' else 'Normal'
        recs.append(record('E-MTAB-13526', lib, re.sub(r'\s+', '', r['Characteristics[individual]']), tissue, hist[r['Description']], fraction, dest,
                           chemistry="10x 3' v3", source_sample=lib, sex=r['Characteristics[sex]'], age=r['Characteristics[age]'], stage_tnm=r['Characteristics[tumor grading]']))
    from joblib import Parallel, delayed
    Parallel(n_jobs=4)(delayed(_emtab_library)(raw, Path(rec['path'])) for rec in recs if not _done(rec['path']))
    return recs, excl


def _emtab_library(raw, dest):
    lib = dest.name
    mat = spio.mmread(str(raw / f'{lib}-matrix.mtx.gz')).tocsr()
    feat = pd.read_csv(raw / f'{lib}-features.tsv.gz', sep='\t', header=None)
    bars = pd.read_csv(raw / f'{lib}-barcodes.tsv.gz', header=None)[0].astype(str).tolist()
    ge = (feat[2] == 'Gene Expression').to_numpy() if feat.shape[1] > 2 else np.ones(len(feat), bool)
    mat, bars = _prefilter(mat[ge, :].tocsc(), bars)
    write_10x(dest, mat, feat.loc[ge, 1].astype(str).tolist(), bars)


# --------------------------------------------------------------------------- GSE154826
GSE154826_FRACTION = {'CD45+ bead enrichment': 'CD45+', 'CD45+ FACS': 'CD45+', 'dead cell depletion kit': 'unsorted', 'CD2+ bead enrichment': 'CD2+'}


def _read_tar_library(path):
    with tarfile.open(path) as tar, tempfile.TemporaryDirectory() as tmp:
        tar.extractall(tmp)
        tmp = Path(tmp)
        mtx = next(tmp.glob('*_matrix.mtx'))
        stem = mtx.name[:-len('_matrix.mtx')]
        feat = pd.read_csv(tmp / f'{stem}_features.tsv', sep='\t', header=None)
        bars = pd.read_csv(tmp / f'{stem}_barcodes.tsv', header=None)[0].astype(str).tolist()
        mat = spio.mmread(str(mtx)).tocsr()
    return mat, feat, bars


def demultiplex(hto_counts, groups):
    """hto_counts: DataFrame cells x HTO names; groups: {group: [HTO names]}.
    Assign a cell to a group if that group's hashtags hold >= HTO_MIN_COUNTS and >= HTO_MIN_FRACTION of all hashtag counts."""
    per_group = pd.DataFrame({g: hto_counts[h].sum(axis=1) for g, h in groups.items()})
    total = per_group.sum(axis=1)
    top = per_group.idxmax(axis=1)
    top_counts = per_group.max(axis=1)
    ok = (top_counts >= HTO_MIN_COUNTS) & (top_counts >= HTO_MIN_FRACTION * total.where(total > 0, np.inf))
    return top.where(ok)


def gse154826(raw, out):
    ann = pd.read_csv(raw / 'GSE154826_sample_annots.csv.gz')
    ann['fraction'] = ann.prep.map(GSE154826_FRACTION)
    human = ann[(ann.Species == 'human') & ann.tissue.isin(['Tumor', 'Normal'])]
    excl = []
    for r in ann[~ann.index.isin(human.index)].itertuples():
        excl.append(dict(cohort='GSE154826', library=f'sample{r.sample_ID}', reason=f'Not lung tissue ({r.tissue})'))
    eligible = human[(human.fraction.isin(['CD45+', 'unsorted'])) & (human.prime == 3)]
    for r in human[~human.index.isin(eligible.index)].itertuples():
        excl.append(dict(cohort='GSE154826', library=f'sample{r.sample_ID}', reason="CD2+ T-cell enrichment" if r.fraction == 'CD2+' else "5' chemistry"))
    pairs = eligible.groupby('patient_ID').tissue.agg(lambda s: set(s))
    paired = set(pairs[pairs.apply(lambda s: s == {'Tumor', 'Normal'})].index)
    for r in eligible[~eligible.patient_ID.isin(paired)].itertuples():
        excl.append(dict(cohort='GSE154826', library=f'sample{r.sample_ID}', reason='No matched tumour/normal partner'))
    eligible = eligible[eligible.patient_ID.isin(paired)]
    jobs = []
    for batch, rows in eligible.groupby('amp_batch_ID'):
        dests = {r.sample_ID: out / f'amp{batch}_sample{r.sample_ID}' for r in rows.itertuples()}
        if not all(_done(d) for d in dests.values()):
            jobs.append((batch, dests))
    from joblib import Parallel, delayed
    Parallel(n_jobs=4)(delayed(_gse154826_batch)(raw, out, batch, ann[ann.amp_batch_ID == batch], dests) for batch, dests in jobs)
    recs = []
    for r in eligible.sort_values(['amp_batch_ID', 'sample_ID']).itertuples():
        dest = out / f'amp{r.amp_batch_ID}_sample{r.sample_ID}'
        if not _done(dest):
            raise RuntimeError(f'GSE154826 library {dest.name} was not written')
        recs.append(record('GSE154826', dest.name, f'P{r.patient_ID}', r.tissue, r.disease, r.fraction, dest,
                           chemistry=f"10x 3' {r.library_chemistry}", source_sample=f'sample{r.sample_ID}', hashed=bool(pd.notna(r.HTO))))
    logs = [json.loads(p.read_text()) for p in sorted(out.glob('amp*_demultiplexing.json'))]
    if logs:
        (out / 'hto_demultiplexing.json').write_text(json.dumps(logs, indent=2, default=str))
    return recs, excl


def _gse154826_batch(raw, out, batch, batch_rows, dests):
    """batch_rows: every annotated sample in the library (including PBMC and mouse spike-ins), so hashtags compete fairly."""
    mat, feat, bars = _read_tar_library(raw / f'GSE154826_amp_batch_ID_{batch}.tar.gz')
    ge = (feat[2] == 'Gene Expression').to_numpy()
    gex = mat[ge, :].tocsc()
    gex, kept = _prefilter(gex, bars)
    kept_idx = pd.Index(bars).get_indexer(kept)
    log = {'amp_batch': int(batch), 'barcodes_prefiltered': len(kept)}
    if batch_rows.HTO.isna().all():
        if len(batch_rows) != 1:
            raise ValueError(f'GSE154826 batch {batch}: multiple samples without hashtags')
        assign = pd.Series(batch_rows.sample_ID.iloc[0], index=kept)
        log['hashed'] = False
    else:
        hto_names = feat[0].astype(str)
        hto_rows = np.flatnonzero(hto_names.str.startswith('HTO_').to_numpy())
        hto = pd.DataFrame(mat.tocsr()[hto_rows, :][:, kept_idx].T.toarray(), index=kept, columns=hto_names.iloc[hto_rows].tolist())
        groups = {}
        for r in batch_rows.itertuples():
            tags = [f'HTO_{t.strip()}' for t in str(r.HTO).split(',')]
            missing = [t for t in tags if t not in hto.columns]
            if missing:
                raise ValueError(f'GSE154826 batch {batch}: hashtags {missing} absent from features')
            groups[r.sample_ID] = tags
        assign = demultiplex(hto, groups)
        human_umi = np.asarray(gex.sum(axis=0)).ravel()
        tissue = batch_rows.set_index('sample_ID').tissue
        log.update(hashed=True, unassigned=int(assign.isna().sum()),
                   unassigned_ge500umi=int((assign.isna().to_numpy() & (human_umi >= 500)).sum()))
        for sid in groups:
            sel = (assign == sid).to_numpy()
            log[f'sample{sid}:{tissue[sid]}'] = {'cells': int(sel.sum()), 'cells_ge500umi': int((sel & (human_umi >= 500)).sum()),
                                                 'median_human_umi': float(np.median(human_umi[sel])) if sel.any() else None}
    genes = feat.loc[ge, 1].astype(str).tolist()
    for sid, dest in dests.items():
        sel = (assign == sid).to_numpy()
        write_10x(dest, gex[:, sel], genes, [b for b, k in zip(kept, sel) if k])
    (out / f'amp{batch}_demultiplexing.json').write_text(json.dumps(log, indent=2, default=str))


ADAPTERS = {'GSE117570': gse117570, 'GSE131907': gse131907, 'GSE123902': gse123902, 'GSE198099': gse198099,
            'E-MTAB-13526': emtab13526, 'GSE154826': gse154826}


def run(cfg, only=None):
    data = Path(cfg['study']['data_dir'])
    all_recs, all_excl = [], []
    for name in cfg['datasets']:
        if only and name not in only:
            continue
        raw = data / 'raw' / name
        if not (raw / 'COMPLETE').exists():
            raise RuntimeError(f'{name}: download incomplete')
        recs, excl = ADAPTERS[name](raw, data / 'prepared' / name)
        print(f'[prepare] {name}: {len(recs)} libraries, {len(excl)} exclusions', flush=True)
        all_recs += recs
        all_excl += excl
    libs = pd.DataFrame(all_recs)
    if libs.library_id.duplicated().any():
        raise ValueError('Duplicated library identifiers')
    prepared = data / 'prepared'
    libs.to_csv(prepared / 'libraries.tsv', sep='\t', index=False)
    pd.DataFrame(all_excl).to_csv(prepared / 'exclusions.tsv', sep='\t', index=False)
    return libs
