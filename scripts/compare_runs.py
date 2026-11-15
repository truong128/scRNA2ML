"""Compare two scRNA2ML results directories, or verify raw downloads against their SHA-256 manifests.

    python scripts/compare_runs.py OLD_RESULTS NEW_RESULTS [--tol 1e-6]
    python scripts/compare_runs.py --verify-raw DATA_RAW
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def flatten(x, prefix=''):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flatten(v, f'{prefix}/{k}'))
        return out
    return {prefix: x}


def same(a, b, tol):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b)) or (np.isnan(a) and np.isnan(b))
    return a == b


def compare_numbers(old, new, tol):
    a = flatten(json.loads((old / 'publication/key_numbers.json').read_text()))
    b = flatten(json.loads((new / 'publication/key_numbers.json').read_text()))
    diff = [(k, a.get(k), b.get(k)) for k in sorted(a.keys() | b.keys()) if k not in a or k not in b or not same(a[k], b[k], tol)]
    print(f'key_numbers.json: {len(a.keys() & b.keys())} shared values, {len(diff)} differ')
    for k, x, y in diff[:50]:
        print(f'  {k}: {x!r} -> {y!r}')
    return len(diff)


def compare_tables(old, new, tol):
    old_t, new_t = old / 'publication/tables', new / 'publication/tables'
    names = sorted({p.name for p in old_t.glob('*.tsv')} | {p.name for p in new_t.glob('*.tsv')})
    bad = 0
    for name in names:
        if not (old_t / name).exists() or not (new_t / name).exists():
            print(f'  {name}: only in {"new" if not (old_t / name).exists() else "old"}')
            bad += 1
            continue
        a = pd.read_csv(old_t / name, sep='\t')
        b = pd.read_csv(new_t / name, sep='\t')
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            print(f'  {name}: shape {a.shape} -> {b.shape}')
            bad += 1
            continue
        num = a.select_dtypes('number').columns.intersection(b.select_dtypes('number').columns)
        other = [c for c in a.columns if c not in num]
        x, y = a[num].to_numpy(float), b[num].to_numpy(float)
        scale = np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))
        gap = np.where(np.isnan(x) & np.isnan(y), 0.0, np.abs(x - y) / scale)
        max_gap = float(np.nanmax(gap)) if gap.size else 0.0
        if np.isnan(gap).any():
            max_gap = float('inf')
        text_diff = int((a[other].astype(str) != b[other].astype(str)).to_numpy().sum())
        if max_gap > tol or text_diff:
            print(f'  {name}: max relative numeric difference {max_gap:.3g}, {text_diff} text cells differ')
            bad += 1
    print(f'tables: {len(names)} compared, {bad} differ')
    return bad


def verify_raw(raw):
    bad = checked = 0
    for manifest in sorted(raw.glob('*/download_manifest.json')):
        for name, rec in json.loads(manifest.read_text()).items():
            h = hashlib.sha256()
            with open(manifest.parent / name, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 22), b''):
                    h.update(chunk)
            checked += 1
            if h.hexdigest() != rec['sha256']:
                print(f'  SHA-256 mismatch: {manifest.parent.name}/{name}')
                bad += 1
    print(f'raw files: {checked} checked against SHA-256 manifests, {bad} mismatched')
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dirs', nargs='*', type=Path)
    p.add_argument('--tol', type=float, default=1e-6)
    p.add_argument('--verify-raw', type=Path)
    a = p.parse_args()
    if a.verify_raw:
        sys.exit(1 if verify_raw(a.verify_raw) else 0)
    if len(a.dirs) != 2:
        p.error('give OLD_RESULTS and NEW_RESULTS')
    old, new = a.dirs
    n = compare_numbers(old, new, a.tol) + compare_tables(old, new, a.tol)
    print('IDENTICAL within tolerance' if n == 0 else f'{n} differences (tolerance {a.tol:g})')
    sys.exit(1 if n else 0)


if __name__ == '__main__':
    main()
