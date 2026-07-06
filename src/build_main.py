#!/usr/bin/env python3
"""Build the main manuscript ``.docx`` from ``output/spec.json``.

Mirrors ``build_supplement.py``: stages the five manuscript figures under the
spec's image keys (``image1.png`` .. ``image5.png``) plus ``authors.json`` and
``refs.bib`` in a temp base dir, builds the document with the global
``docx-tools`` CLI, then injects the reviewer comments to produce the versioned,
comment-annotated deliverable (``output/draft_v{VERSION}.docx``).

The main paper and the SI are separate documents; this script touches only the
main paper.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT / "output/spec.json"
PAPER_DIR = PROJECT / "output"
OUTPUT_DIR = PROJECT / "output"
AUTHORS = PROJECT / "authors.json"
REFS = PROJECT / "references.bib"

VERSION = "1.4"
COMMENTS = OUTPUT_DIR / f"draft_v{VERSION}_comments.json"

# spec image key -> source figure file
FIGURES = {
    "image1.png": PAPER_DIR / "Fig_1.png",
    "image2.png": PAPER_DIR / "Fig_2.png",
    "image3.png": PAPER_DIR / "Fig_3.png",
    "image4.png": PAPER_DIR / "Fig_4.png",
    "image5.png": PAPER_DIR / "Fig_5.png",
}


def main() -> None:
    # `--submission` builds the clean, comment-free file for journal upload
    # (draft_v{VERSION}_submission.docx); the default build injects the
    # coauthor-review comments (draft_v{VERSION}.docx).
    submission = "--submission" in sys.argv

    for key, src in FIGURES.items():
        if not src.exists():
            raise FileNotFoundError(f"figure not found for {key}: {src}")
    required = [AUTHORS, REFS] if submission else [AUTHORS, REFS, COMMENTS]
    for req in required:
        if not req.exists():
            raise FileNotFoundError(f"required input not found: {req}")

    suffix = "_submission" if submission else ""
    output = OUTPUT_DIR / f"draft_v{VERSION}{suffix}.docx"
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for key, src in FIGURES.items():
            shutil.copy(src, tmp_path / key)
        shutil.copy(AUTHORS, tmp_path / "authors.json")
        shutil.copy(REFS, tmp_path / "refs.bib")

        base = tmp_path / "base.docx"
        subprocess.run(
            ["docx-tools", "build", str(SPEC_PATH),
             "-o", str(base), "--base-dir", str(tmp_path)],
            check=True,
        )
        if submission:
            shutil.copy(base, output)  # clean: no comment injection
        else:
            subprocess.run(
                ["docx-tools", "inject", str(base),
                 "--comments", str(COMMENTS), "-o", str(output)],
                check=True,
            )
    print(f"[BUILT] {output}")


if __name__ == "__main__":
    main()
