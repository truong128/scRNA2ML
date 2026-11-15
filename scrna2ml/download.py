"""Resumable, size-verified, parallel downloads of public scRNA-seq count matrices."""
import hashlib
import json
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import load_config

_LOCK = threading.Lock()
# Public archives throttle aggressive clients (NCBI returns HTTP 503); cap concurrency per host.
_HOST_LIMITS = {'ftp.ncbi.nlm.nih.gov': threading.Semaphore(3), 'ftp.ebi.ac.uk': threading.Semaphore(6)}


def _host_lock(url):
    return _HOST_LIMITS.get(urllib.parse.urlparse(url).netloc, threading.Semaphore(4))


def _with_retry(fn, attempts=8):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(min(60, 5 * 2 ** i))


def _listing(url, pattern):
    html = _with_retry(lambda: urllib.request.urlopen(url, timeout=120).read().decode())
    names = sorted(set(re.findall(r'href="(' + pattern + r')"', html)))
    if not names:
        raise RuntimeError(f'No files matching {pattern} at {url}')
    return [url + n for n in names]


def _remote_size(url):
    def head():
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=120) as r:
            return int(r.headers['Content-Length'])
    return _with_retry(head)


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def resolve_urls(spec):
    if spec['source'] == 'geo':
        return list(spec['files'])
    if spec['source'] in ('geo_listing', 'biostudies_listing'):
        return _listing(spec['listing'], spec['pattern'])
    raise ValueError(f"Unknown source {spec['source']}")


def _manifest_path(target):
    return target / 'download_manifest.json'


def _read_manifest(target):
    p = _manifest_path(target)
    return json.loads(p.read_text()) if p.exists() else {}


def _fetch(name, url, target, retries=8):
    dest = target / url.rsplit('/', 1)[1]
    with _LOCK:
        done = _read_manifest(target).get(dest.name)
    if done and dest.exists() and dest.stat().st_size == done['bytes']:
        return dest.name
    with _host_lock(url):
        expected = _remote_size(url)
        for i in range(retries):
            subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '5', '--retry-delay', '10', '-C', '-', '-o', str(dest), url], check=False)
            if dest.exists() and dest.stat().st_size == expected:
                # A resumed transfer can have the right size but corrupt content; gzip CRC catches it.
                if not dest.name.endswith('.gz') or subprocess.run(['gzip', '-t', str(dest)], capture_output=True).returncode == 0:
                    break
                print(f'[download] {name}: {dest.name} failed gzip integrity check; re-downloading', flush=True)
                dest.unlink()
            if dest.exists() and dest.stat().st_size > expected:
                dest.unlink()
            time.sleep(min(60, 5 * 2 ** i))
        else:
            raise RuntimeError(f'Download failed or size mismatch: {url}')
    digest = _sha256(dest)
    with _LOCK:
        manifest = _read_manifest(target)
        manifest[dest.name] = {'url': url, 'bytes': expected, 'sha256': digest}
        _manifest_path(target).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f'[download] {name}: {dest.name} ({expected / 1e6:.1f} MB)', flush=True)
    return dest.name


def run(config_path, only=None, workers=8):
    cfg = load_config(config_path)
    jobs = []
    for name, spec in cfg['datasets'].items():
        if only and name not in only:
            continue
        target = Path(cfg['study']['data_dir']) / 'raw' / name
        target.mkdir(parents=True, exist_ok=True)
        urls = resolve_urls(spec)
        jobs.extend((name, url, target) for url in urls)
        (target / 'expected_files.json').write_text(json.dumps([u.rsplit('/', 1)[1] for u in urls], indent=2))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, *job): job for job in jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                failures.append((futures[fut][1], str(e)))
                print(f'[download] FAILED {futures[fut][1]}: {e}', flush=True)
    for name in {j[0] for j in jobs}:
        target = Path(cfg['study']['data_dir']) / 'raw' / name
        expected = set(json.loads((target / 'expected_files.json').read_text()))
        have = set(_read_manifest(target))
        if expected <= have:
            (target / 'COMPLETE').write_text(json.dumps({'files': len(expected)}))
    if failures:
        raise RuntimeError(f'{len(failures)} downloads failed')
