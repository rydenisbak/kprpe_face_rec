"""KP-RPE ViT-B weights (CVLface, WebFace12M): where FaceRecognizer looks for them and how they are fetched.

Lookup order for FaceRecognizer(weights=None):
  1. kprpe_face/weights/model.safetensors next to the code, if someone put the file there;
  2. the cache: $KPRPE_FACE_CACHE, else $XDG_CACHE_HOME/kprpe_face, else ~/.cache/kprpe_face. If the file is not there,
     it is downloaded on first use from the authors' Hugging Face repo (pinned revision; size and sha256 checked).
     $KPRPE_FACE_URL overrides the URL (a mirror of the same file).

    kprpe-face-download                   # console script installed by pip: pre-download into the cache
    python -m kprpe_face.download --out PATH

The weights are not part of this package and not covered by its license. They are published without a license of
their own; the model card asks to follow the license of the training data (WebFace260M / WebFace12M: academic
research only, no commercial use).
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.request

REPO = 'minchul/cvlface_adaface_vit_base_kprpe_webface12m'
REVISION = 'daefd5012d369588bd214fbaf4cc6b1d286e7066'  # pinned commit of the Hugging Face repo
URL = f'https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors'
SHA256 = '99d16ed4aac0fdf0fcc82526b9b70703f3ec8c3041bf1bf44bd22751536e65db'
SIZE = 460344344
FILENAME = f'kprpe_vitb_webface12m_{SHA256[:8]}.safetensors'
PACKAGE_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'model.safetensors')
CHUNK = 1 << 20

NOTICE = (f'KP-RPE weights: https://huggingface.co/{REPO} (revision {REVISION[:8]}, {SIZE / 1e6:.0f} MB). They have '
          'no license of their own; the model card asks to follow the training data license (WebFace260M: academic '
          'research only, no commercial use). Use them for non-commercial research only.')


def cache_dir():
    if os.environ.get('KPRPE_FACE_CACHE'):
        return os.environ['KPRPE_FACE_CACHE']
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'kprpe_face')


def cache_path():
    return os.path.join(cache_dir(), FILENAME)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def download(path, url=None):
    """Download the weights to path (temp file + atomic rename), check size and sha256. Returns path."""
    url = url or os.environ.get('KPRPE_FACE_URL') or URL
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = f'{path}.{os.getpid()}.part'
    print(f'{NOTICE}\ndownloading {url}\n         -> {path}', file=sys.stderr)
    h, done, t0, shown = hashlib.sha256(), 0, time.time(), 0.0
    req = urllib.request.Request(url, headers={'User-Agent': 'kprpe_face-download/0.1'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(part, 'wb') as f:
            total = int(r.headers.get('Content-Length') or SIZE)
            while True:
                block = r.read(CHUNK)
                if not block:
                    break
                f.write(block)
                h.update(block)
                done += len(block)
                now = time.time()
                if now - shown > 1 or done == total:
                    shown = now
                    sys.stderr.write(f'\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  '
                                     f'{done / 1e6 / max(now - t0, 1e-6):6.1f} MB/s')
        sys.stderr.write('\n')
        if done != SIZE or h.hexdigest() != SHA256:
            raise RuntimeError(f'downloaded file is not the expected one: {done} bytes, sha256 {h.hexdigest()} '
                               f'(expected {SIZE} bytes, sha256 {SHA256})')
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path


def resolve(download_missing=True):
    """Path of the weights: next to the code if present, else the cache (downloaded there if missing)."""
    if os.path.exists(PACKAGE_WEIGHTS):
        return PACKAGE_WEIGHTS
    path = cache_path()
    if os.path.exists(path):
        return path
    if not download_missing:
        raise FileNotFoundError(f'KP-RPE weights not found in {PACKAGE_WEIGHTS} or {path}; '
                                f'run kprpe-face-download, or pass weights=PATH')
    return download(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Download the KP-RPE ViT-B weights (sha256-checked).')
    ap.add_argument('--out', help=f'target file (default: {cache_path()})')
    ap.add_argument('--url', help='mirror URL of the same file')
    args = ap.parse_args(argv)
    if args.out:
        path = os.path.abspath(args.out)
    elif os.path.exists(PACKAGE_WEIGHTS):
        path = PACKAGE_WEIGHTS
    else:
        path = cache_path()
    if os.path.exists(path):
        if sha256_file(path) == SHA256:
            print(f'already present, sha256 OK: {path}')
            return 0
        print(f'existing file has a different sha256, downloading again: {path}', file=sys.stderr)
    try:
        download(path, url=args.url)
    except Exception as e:  # network or checksum problem: report and fail
        print(f'ERROR: {e}', file=sys.stderr)
        return 1
    print(f'saved, sha256 OK: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
