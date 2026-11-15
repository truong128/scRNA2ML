"""Configuration loading; relative paths resolve against the repository root."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    cfg = yaml.safe_load(path.read_text())
    for key in ('data_dir', 'results_dir'):
        p = Path(cfg['study'][key])
        cfg['study'][key] = str(p if p.is_absolute() else REPO / p)
    cfg['config_path'] = str(path)
    return cfg
