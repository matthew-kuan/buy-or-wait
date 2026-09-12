"""Build code.zip at the repository root for submission.

    python code/package.py

Includes code/ (with evaluation/ and its usage_report.md), the response cache .cache/ (so a
re-run needs no API key), requirements.txt, .env.example and README.md. Excludes log.txt, .env,
dataset/, output.csv, git metadata and __pycache__.
"""
from __future__ import annotations

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "code.zip")
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def add_tree(z: zipfile.ZipFile, base: str, arc_prefix: str) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in sorted(filenames):
            if fn.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, fn)
            arc = os.path.join(arc_prefix, os.path.relpath(full, base)).replace(os.sep, "/")
            z.write(full, arc)
            n += 1
    return n


def main() -> int:
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = add_tree(z, HERE, "code")
        cache = os.path.join(ROOT, ".cache")
        if os.path.isdir(cache):
            n += add_tree(z, cache, ".cache")
        for fn in ("requirements.txt", ".env.example"):
            p = os.path.join(ROOT, fn)
            if os.path.exists(p):
                z.write(p, fn)
                n += 1
        readme = os.path.join(HERE, "README.md")
        if os.path.exists(readme):
            z.write(readme, "README.md")
            n += 1
    print(f"wrote {OUT} ({n} files, {os.path.getsize(OUT):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
