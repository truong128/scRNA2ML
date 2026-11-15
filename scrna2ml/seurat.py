"""Run the label-blind per-library Seurat QC/annotation/pseudobulk stage and collect a QC audit table."""
import json
import subprocess
from pathlib import Path

import pandas as pd

R_SCRIPT = Path(__file__).parent / 'R' / 'seurat_qc.R'


def run(cfg):
    data = Path(cfg['study']['data_dir'])
    out = Path(cfg['study']['results_dir']) / 'seurat'
    out.mkdir(parents=True, exist_ok=True)
    libs = pd.read_csv(data / 'prepared' / 'libraries.tsv', sep='\t')
    manifest = pd.DataFrame({'sample_id': libs.library_id,
                             'matrix': [str(Path(p).resolve() / 'matrix.mtx.gz') for p in libs.path],
                             'genes': [str(Path(p).resolve() / 'genes.tsv.gz') for p in libs.path],
                             'barcodes': [str(Path(p).resolve() / 'barcodes.tsv.gz') for p in libs.path]})
    for col in ('matrix', 'genes', 'barcodes'):
        missing = [p for p in manifest[col] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} prepared files missing, e.g. {missing[0]}')
    manifest.to_csv(out / 'manifest.tsv', sep='\t', index=False)
    (out / 'qc.json').write_text(json.dumps(cfg['qc'], indent=2))
    subprocess.run(['Rscript', str(R_SCRIPT), str(out / 'manifest.tsv'), str(out), str(out / 'qc.json')], check=True)
    return audit(cfg)


def audit(cfg):
    data = Path(cfg['study']['data_dir'])
    out = Path(cfg['study']['results_dir']) / 'seurat'
    libs = pd.read_csv(data / 'prepared' / 'libraries.tsv', sep='\t')
    rows = []
    for lib in libs.itertuples():
        p = out / lib.library_id / 'complete.json'
        if not p.exists():
            raise FileNotFoundError(f'Seurat stage incomplete for {lib.library_id}')
        info = json.loads(p.read_text())
        rows.append({'library_id': lib.library_id, 'status': info['status'], 'input_cells': info.get('input_cells'),
                     'retained_cells': info.get('retained'), 'immune_cells': info.get('immune_cells'),
                     'genes': info.get('genes'), 'median_genes_per_cell': info.get('median_genes_per_cell'),
                     'median_umis_per_cell': info.get('median_umis_per_cell'), 'median_percent_mt': info.get('median_percent_mt'),
                     'unknown_fraction': info.get('unknown_fraction'), 'high_count_flagged': info.get('high_count_flagged')})
    qc = libs.merge(pd.DataFrame(rows), on='library_id', validate='one_to_one')
    qc.to_csv(out / 'qc_audit.tsv', sep='\t', index=False)
    return qc
