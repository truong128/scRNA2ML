"""End-to-end pipeline: python -m scrna2ml run --config config/nsclc.yaml [--stages download,prepare,...]"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .config import load_config

STAGES = ['download', 'prepare', 'seurat', 'features', 'ml', 'analysis', 'report']


def stage_download(cfg, args):
    from . import download
    download.run(cfg['config_path'], workers=args.download_workers)


def stage_prepare(cfg, args):
    from . import adapters
    adapters.run(cfg)


def stage_seurat(cfg, args):
    from . import seurat
    seurat.run(cfg)


def stage_features(cfg, args):
    from . import features
    for comp in cfg['features']['compartments']:
        features.build(cfg, comp)


def schemes_for(cfg, comp):
    from . import ml
    out = []
    for s in cfg['ml']['schemes'].get(comp, []):
        if s == 'within':
            units = ml.load(cfg, comp)[0]
            sizes = units.groupby('cohort').donor_id.nunique()
            out += [f'within::{c}' for c, n in sizes.items() if n >= cfg['ml']['within_min_donors']]
        else:
            out.append(s)
    return out


def stage_ml(cfg, args):
    from . import ml
    for comp in cfg['features']['compartments']:
        for scheme in schemes_for(cfg, comp):
            ml.run_scheme(cfg, comp, scheme, n_jobs=cfg['ml']['n_jobs'])
        for scheme in cfg['ml']['permutation_schemes'].get(comp, []):
            ml.permutation_test(cfg, comp, scheme, n_jobs=cfg['ml']['n_jobs'])
        ml.final_model(cfg, comp)


def stage_analysis(cfg, args):
    from . import analysis
    for comp in cfg['features']['compartments']:
        analysis.paired_meta(cfg, comp)
        analysis.cohort_predictability(cfg, comp)
        analysis.feature_set_ablation(cfg, comp, 'loco')
        analysis.subgroups(cfg, comp, 'loco')
        analysis.stability(cfg, comp, 'loco')
        analysis.fold_local_panel(cfg, comp)
        analysis.technical_control(cfg, comp)


def stage_report(cfg, args):
    from . import report
    report.run(cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog='scrna2ml', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--config', required=True)
    r.add_argument('--stages', default=','.join(STAGES), help='comma-separated subset of ' + ','.join(STAGES))
    r.add_argument('--download-workers', type=int, default=8)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        p.error(f'unknown stages {sorted(unknown)}')
    log = Path(cfg['study']['results_dir']) / 'run_log.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    provenance = {'config_sha256': hashlib.sha256(Path(cfg['config_path']).read_bytes()).hexdigest(), 'code_sha256': code_hash(),
                  'git_commit': git_commit()}
    for stage in [s for s in STAGES if s in stages]:
        t0 = time.time()
        print(f'=== stage {stage} ===', flush=True)
        globals()[f'stage_{stage}'](cfg, args)
        with open(log, 'a') as f:
            f.write(json.dumps({'stage': stage, 'seconds': round(time.time() - t0, 1), 'version': __version__, **provenance,
                                'python': sys.version.split()[0], 'platform': platform.platform(), 'pandas': pd.__version__,
                                'finished': time.strftime('%Y-%m-%dT%H:%M:%S')}) + '\n')


def code_hash():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*')):
        if p.suffix in ('.py', '.R'):
            h.update(p.relative_to(Path(__file__).parent).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
        return out + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == '__main__':
    main()
