"""Tests for GitHub skill install, search, and skill.json format."""
import json
import tempfile
from pathlib import Path

from ultra.core.skills import Skill, SkillManager


def _make_skill_dir(skills_dir: Path, name: str, skill_md: str = None,
                    skill_json: dict = None) -> Path:
    folder = skills_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    if skill_md:
        (folder / "SKILL.md").write_text(skill_md)
    if skill_json:
        (folder / "skill.json").write_text(json.dumps(skill_json))
    return folder


def test_load_skill_md_only():
    d = Path(tempfile.mkdtemp())
    _make_skill_dir(d, "test-skill", skill_md="---\ndescription: A test skill\ntriggers: test, demo\n---\nSkill body here.")
    sm = SkillManager(d)
    skills = sm.list()
    assert len(skills) == 1
    assert skills[0].name == "test-skill"
    assert skills[0].description == "A test skill"
    assert "test" in skills[0].triggers
    assert "Skill body here" in skills[0].body


def test_load_skill_json_only():
    d = Path(tempfile.mkdtemp())
    _make_skill_dir(d, "json-skill", skill_json={
        "name": "json-skill",
        "description": "A JSON-only skill",
        "version": "1.0.0",
        "author": "coder1317",
        "tags": ["hardware", "esp32"],
        "triggers": ["esp32", "pcb"],
    })
    sm = SkillManager(d)
    skills = sm.list()
    assert len(skills) == 1
    assert skills[0].version == "1.0.0"
    assert skills[0].author == "coder1317"
    assert "hardware" in skills[0].tags


def test_load_skill_both():
    d = Path(tempfile.mkdtemp())
    _make_skill_dir(d, "both", 
        skill_md="---\ndescription: From markdown\ntriggers: md-trigger\n---\nBody from MD.",
        skill_json={"version": "2.0.0", "author": "test", "tags": ["web"]})
    sm = SkillManager(d)
    s = sm.get("both")
    assert s is not None
    assert s.description == "From markdown"  # SKILL.md wins
    assert s.version == "2.0.0"  # skill.json supplements
    assert s.author == "test"
    assert "Body from MD" in s.body


def test_skill_prompt_fragment():
    s = Skill(name="test", description="A test skill", version="1.0",
              body="Body content here")
    frag = s.prompt_fragment
    assert "[SKILL: test]" in frag
    assert "A test skill" in frag
    assert "v1.0" in frag
    assert "Body content here" in frag


def test_skill_to_dict():
    s = Skill(name="x", description="desc", tags=["a"], source_repo="owner/repo")
    d = s.to_dict()
    assert d["name"] == "x"
    assert d["source_repo"] == "owner/repo"
    assert "a" in d["tags"]


def test_skill_for_request():
    d = Path(tempfile.mkdtemp())
    _make_skill_dir(d, "hw", skill_md="---\ndescription: hardware\ntriggers: arduino, esp32\n---\nHW body")
    _make_skill_dir(d, "web", skill_md="---\ndescription: web\ntriggers: flask, fastapi\n---\nWeb body")
    sm = SkillManager(d)
    matched = sm.for_request("build an arduino project")
    assert len(matched) == 1
    assert matched[0].name == "hw"
    matched = sm.for_request("build a flask api")
    assert len(matched) == 1
    assert matched[0].name == "web"


def test_skill_context():
    d = Path(tempfile.mkdtemp())
    _make_skill_dir(d, "s1", skill_md="---\ndescription: Skill 1\ntriggers: test\n---\nBody 1")
    sm = SkillManager(d)
    ctx = sm.context("test this")
    assert "SKILL: s1" in ctx
    assert "Body 1" in ctx


def test_uninstall():
    d = Path(tempfile.mkdtemp())
    _make_skill_dir(d, "removable", skill_md="---\ndescription: to remove\n---\n")
    sm = SkillManager(d)
    assert len(sm.list()) == 1
    assert sm.uninstall("removable")
    assert len(sm.list()) == 0
    assert not sm.uninstall("removable")  # already gone


def test_cache_invalidation():
    d = Path(tempfile.mkdtemp())
    sm = SkillManager(d)
    assert len(sm.list()) == 0
    _make_skill_dir(d, "new", skill_md="---\ndescription: new\n---\n")
    sm._cache = None  # simulate invalidation
    assert len(sm.list()) == 1
