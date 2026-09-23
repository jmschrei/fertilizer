"""Install the bundled fertilizer Agent Skill for Claude Code.

Claude Code does not scan `site-packages` for skills, so the skill ships as
package data under `skills/fertilizer/` and `fertilizer install-skill` copies
it into `~/.claude/skills/fertilizer`. Copying is opt-in via this command and
never happens on import or install.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

__all__ = ["SKILL_NAME", "bundled_skill_dir", "install_skill", "run_install_skill"]

SKILL_NAME = "fertilizer"


def bundled_skill_dir() -> Path:
    """Return the directory holding the bundled `SKILL.md` and `references/`."""
    return Path(__file__).resolve().parent / "skills" / SKILL_NAME


def install_skill(
    directory: str | Path | None = None, force: bool = False, symlink: bool = False,
) -> Path:
    """Copy (or symlink) the bundled skill into a Claude Code skills directory.

    The skill is written to `<directory>/fertilizer`, where `directory`
    defaults to `~/.claude/skills`. An existing destination raises
    `FileExistsError` unless `force` is set, in which case it is removed
    first so the installed copy matches the package exactly. With `symlink`,
    the destination points at the package's own copy instead, so edits to an
    editable install show up without reinstalling.
    """
    source = bundled_skill_dir()
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(
            f"bundled skill not found at {source}; the package may be "
            "installed without its data files"
        )

    skills_dir = (
        Path(directory).expanduser() if directory is not None
        else Path.home() / ".claude" / "skills"
    )
    dest = skills_dir / SKILL_NAME

    if dest.is_symlink() or dest.exists():
        if not force:
            raise FileExistsError(
                f"{dest} already exists. Re-run with --force to overwrite it."
            )
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    skills_dir.mkdir(parents=True, exist_ok=True)
    if symlink:
        dest.symlink_to(source, target_is_directory=True)
    else:
        shutil.copytree(
            source, dest,
            ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"),
        )
    return dest


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the `fertilizer install-skill` subcommand."""
    parser = subparsers.add_parser(
        "install-skill",
        help="Install the bundled fertilizer agent skill for Claude Code.",
    )
    parser.add_argument(
        "-d", "--directory", default=None, metavar="DIR",
        help="Skills directory to install into (default: ~/.claude/skills).",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Symlink the packaged skill instead of copying it. Reflects "
             "in-place edits, but breaks if the package moves.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite an existing installation at the destination.",
    )
    parser.set_defaults(func=run_install_skill)
    return parser


def run_install_skill(args: argparse.Namespace) -> int:
    dest = install_skill(args.directory, force=args.force, symlink=args.symlink)
    verb = "Symlinked" if args.symlink else "Installed"
    print(f"{verb} fertilizer skill to {dest}")
    print("Restart Claude Code (or reload skills) to pick it up.")
    return 0
