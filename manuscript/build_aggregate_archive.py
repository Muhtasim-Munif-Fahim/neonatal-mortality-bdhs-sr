"""Build the DOI-ready, code-only-compatible aggregate reproducibility archive."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "manuscript" / "scientific_reports"
SRC = ROOT / "results" / "aggregate_release"
OUT = SR / "aggregate_archive"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    expected_parent = (ROOT / "manuscript" / "scientific_reports").resolve()
    if OUT.resolve().parent != expected_parent:
        raise RuntimeError(f"refusing to rebuild unexpected archive path: {OUT}")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "aggregate_tables").mkdir(parents=True)
    (OUT / "reporting_checklists").mkdir()
    for p in SRC.iterdir():
        if p.is_file():
            shutil.copy2(p, OUT / "aggregate_tables" / p.name)
    for p in (SR / "checklists").glob("*.md"):
        shutil.copy2(p, OUT / "reporting_checklists" / p.name)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (OUT / "analysis_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (OUT / "reproduction.txt").write_text(
        "Repository: https://github.com/Muhtasim-Munif-Fahim/neonatal-mortality-bdhs-sr\n"
        f"Commit: {commit}\n"
        "Environment: create an isolated environment from requirements.txt\n"
        "Authorised-data command: python run_all.py --force\n"
        "Integrity command: python -m src.checks\n"
        "The four controlled-access DHS Births Recode archives must be supplied "
        "by an authorised user; they are not included here.\n",
        encoding="utf-8")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    manifest = {
        "title": "Aggregate reproducibility materials for temporal neonatal-mortality evaluation in Bangladesh",
        "created": str(date.today()),
        "analysis_commit": commit,
        "deposit_status": "DOI/URL pending user deposit",
        "exclusions": ["DHS microdata", "record-level predictions", "identifiers", "serialized fitted model"],
        "files": [str(p.relative_to(OUT)).replace("\\", "/") for p in files],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    checksums = "\n".join(
        f"{sha256(p)}  {str(p.relative_to(OUT)).replace(chr(92), '/')}" for p in files
    ) + "\n"
    (OUT / "checksums.sha256").write_text(checksums, encoding="utf-8")

    archive = SR / "aggregate_reproducibility_archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(x for x in OUT.rglob("*") if x.is_file()):
            zf.write(p, p.relative_to(OUT))
    print(archive)


if __name__ == "__main__":
    main()
