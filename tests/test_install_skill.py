"""Tests for `fertilizer install-skill` and the skill data it ships."""

from __future__ import annotations

import re

import pytest

from fertilizer.cli import main
from fertilizer.install_skill import SKILL_NAME, bundled_skill_dir, install_skill

SKILL_DIR = bundled_skill_dir()


def _documents():
	paths = [SKILL_DIR / "SKILL.md", *sorted((SKILL_DIR / "references").glob("*.md"))]
	return [(p, p.read_text()) for p in paths]


class TestInstall:
	def test_copy_installs_skill_and_references(self, tmp_path):
		dest = install_skill(tmp_path)
		assert dest == tmp_path / SKILL_NAME
		assert (dest / "SKILL.md").is_file()
		assert sorted(p.name for p in (dest / "references").glob("*.md")) == sorted(
		    p.name for p in (SKILL_DIR / "references").glob("*.md")
		)

	def test_default_directory_is_under_home(self, tmp_path, monkeypatch):
		monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
		dest = install_skill()
		assert dest == tmp_path / ".claude" / "skills" / SKILL_NAME
		assert (dest / "SKILL.md").is_file()

	def test_existing_destination_without_force_raises(self, tmp_path):
		install_skill(tmp_path)
		with pytest.raises(FileExistsError, match="--force"):
			install_skill(tmp_path)

	def test_force_replaces_stale_content(self, tmp_path):
		dest = install_skill(tmp_path)
		stray = dest / "stray.md"
		stray.write_text("stale")
		install_skill(tmp_path, force=True)
		assert not stray.exists()
		assert (dest / "SKILL.md").is_file()

	def test_excludes_checkpoints_and_pycache(self, tmp_path, monkeypatch):
		src = tmp_path / "src" / SKILL_NAME
		(src / "references" / ".ipynb_checkpoints").mkdir(parents=True)
		(src / "__pycache__").mkdir()
		(src / "SKILL.md").write_text("---\nname: x\n---\n")
		monkeypatch.setattr("fertilizer.install_skill.bundled_skill_dir", lambda: src)
		dest = install_skill(tmp_path / "out")
		assert not (dest / "references" / ".ipynb_checkpoints").exists()
		assert not (dest / "__pycache__").exists()

	def test_missing_bundled_skill_raises(self, tmp_path, monkeypatch):
		monkeypatch.setattr(
		    "fertilizer.install_skill.bundled_skill_dir", lambda: tmp_path / "nope"
		)
		with pytest.raises(FileNotFoundError, match="bundled skill not found"):
			install_skill(tmp_path)

	def test_symlink_points_at_package_copy(self, tmp_path):
		dest = install_skill(tmp_path, symlink=True)
		assert dest.is_symlink()
		assert dest.resolve() == SKILL_DIR
		install_skill(tmp_path, force=True)
		assert not dest.is_symlink()

	def test_cli_install_and_collision_exit_code(self, tmp_path, capsys):
		assert main(["install-skill", "-d", str(tmp_path)]) == 0
		assert "Installed fertilizer skill" in capsys.readouterr().out
		assert main(["install-skill", "-d", str(tmp_path)]) == 2
		assert "Re-run with --force" in capsys.readouterr().err
		assert main(["install-skill", "-d", str(tmp_path), "--force"]) == 0


class TestSkillData:
	def test_frontmatter_has_name_and_description(self):
		text = (SKILL_DIR / "SKILL.md").read_text()
		match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
		assert match, "SKILL.md must start with YAML frontmatter"
		fields = dict(
		    line.split(":", 1) for line in match.group(1).splitlines() if ":" in line
		)
		assert fields["name"].strip() == SKILL_NAME
		description = fields["description"].strip()
		assert 0 < len(description) <= 1024

	def test_every_reference_is_linked_from_the_router(self):
		router = (SKILL_DIR / "SKILL.md").read_text()
		for path in sorted((SKILL_DIR / "references").glob("*.md")):
			assert f"`references/{path.name}`" in router, (
			    f"references/{path.name} is not linked from SKILL.md"
			)

	def test_every_md_mention_is_a_full_path_that_resolves(self):
		mention = re.compile(r"`([A-Za-z0-9_/.-]*\.md)`")
		for path, text in _documents():
			for target in mention.findall(text):
				assert target == "SKILL.md" or target.startswith("references/"), (
				    f"{path.name} names `{target}` without its references/ prefix"
				)
				assert (SKILL_DIR / target).is_file(), (
				    f"{path.name} points at `{target}`, which does not exist"
				)
