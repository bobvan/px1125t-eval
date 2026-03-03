#!/usr/bin/env python3
"""
check_shared.py — Compare files shared between rigs and file a bead on divergence.

Reads shared_files.toml from the repo root, locates each source file in the
source rig's Gas Town mayor clone, and compares content with the local copy.
When a file has diverged, creates a bd bead so the discrepancy gets reviewed.

Usage:
    python3 scripts/check_shared.py [--gt-root ~/gt] [--dry-run]

Intended as:
  - A git post-commit hook in both repos  (install via scripts/install-hooks.sh)
  - A Deacon plugin (plugins/check-shared-code/plugin.md) run on a daily cron

Exit codes:
    0  all files match (or no shared files configured)
    1  one or more files diverged (bead(s) filed)
    2  configuration or runtime error
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

try:
    import tomllib                     # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib        # pip install tomli
    except ImportError:
        sys.exit("ERROR: tomllib not available; install tomli: pip install tomli")

REPO_ROOT = Path(__file__).parent.parent
MANIFEST  = REPO_ROOT / "shared_files.toml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def bead_exists(subject_fragment: str) -> bool:
    """Return True if an open bead with this subject fragment already exists."""
    try:
        result = subprocess.run(
            ["bd", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        return subject_fragment.lower() in result.stdout.lower()
    except Exception:
        return False


def file_bead(local: str, source_rig: str, source_path: str,
              dry_run: bool, gt_root: Path) -> None:
    subject = f"Shared file diverged: {local} vs {source_rig}/{source_path}"
    body = (
        f"The file `{local}` in px1125t_eval has diverged from its source "
        f"`{source_path}` in the `{source_rig}` rig.\n\n"
        f"Review the diff and either:\n"
        f"  - Sync one direction if the change should propagate\n"
        f"  - Fork intentionally and remove from shared_files.toml\n\n"
        f"To diff:\n"
        f"  diff ~/gt/{source_rig}/mayor/rig/{source_path} "
        f"~/gt/px1125t_eval/mayor/rig/{local}"
    )
    if dry_run:
        print(f"  [dry-run] would create bead: {subject}")
        return
    if bead_exists(local):
        print(f"  (bead already open for {local}, skipping)")
        return
    # bd needs to run against the rig's beads database, not the source repo.
    # Discover it via the Gas Town rig clone (gt_root / rig_name / .beads).
    beads_dir = gt_root / "px1125t_eval" / ".beads"
    env = {**__import__("os").environ, "BEADS_DIR": str(beads_dir)}
    result = subprocess.run(
        ["bd", "create", subject, "-t", "task",
         "--description", body],
        check=False, env=env, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Bead filed: {subject}")
    else:
        print(f"  WARNING: bd create failed: {result.stderr.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Check shared files between rigs")
    ap.add_argument("--gt-root", default=str(Path.home() / "gt"),
                    help="Gas Town root directory (default: ~/gt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report divergences but do not create beads")
    args = ap.parse_args()

    gt_root = Path(args.gt_root).expanduser()

    if not MANIFEST.exists():
        print(f"No shared_files.toml at {MANIFEST} — nothing to check.")
        return 0

    with open(MANIFEST, "rb") as f:
        config = tomllib.load(f)

    entries = config.get("shared", [])
    if not entries:
        print("shared_files.toml has no [[shared]] entries.")
        return 0

    diverged = 0
    for entry in entries:
        local_rel   = entry["local"]
        source_rig  = entry["source_rig"]
        source_path = entry["source_path"]

        local_file  = REPO_ROOT / local_rel
        source_file = gt_root / source_rig / "mayor" / "rig" / source_path

        if not local_file.exists():
            print(f"  MISSING local  : {local_rel}")
            diverged += 1
            file_bead(local_rel, source_rig, source_path, args.dry_run, gt_root)
            continue

        if not source_file.exists():
            print(f"  MISSING source : {source_rig}/{source_path}  "
                  f"(rig clone not found at {source_file})")
            continue   # source rig may not be cloned on this machine

        local_hash  = sha256(local_file)
        source_hash = sha256(source_file)

        if local_hash == source_hash:
            print(f"  OK      {local_rel}")
        else:
            print(f"  DIVERGED {local_rel}  ← {source_rig}/{source_path}")
            diverged += 1
            file_bead(local_rel, source_rig, source_path, args.dry_run, gt_root)

    if diverged:
        print(f"\n{diverged} file(s) diverged.")
        return 1

    print(f"All {len(entries)} shared file(s) match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
