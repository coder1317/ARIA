import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.config import Config
from ultra.tools.editor import Editor, REPLACE_BLOCK
from ultra.tools.terminal import Terminal, check_safety


def test_blocklist():
    cfg = Config()
    assert check_safety("rm -rf /", cfg) is not None
    assert check_safety("echo hello", cfg) is None


def test_terminal_run():
    cfg = Config(command_timeout=10)
    term = Terminal(cfg)
    result = term.run("echo hello")
    assert result.ok
    assert "hello" in result.output


def test_terminal_blocked():
    cfg = Config()
    term = Terminal(cfg)
    result = term.run("rm -rf /")
    assert result.blocked


def test_editor_write_and_replace(tmp_path):
    f = tmp_path / "app.py"
    Editor.write_file(str(f), "x = 1\n")
    assert "x = 1" in f.read_text()

    res = Editor.apply_replace(str(f), "x = 1", "x = 2")
    assert res.ok
    assert "x = 2" in f.read_text()

    # unknown block -> not found
    res2 = Editor.apply_replace(str(f), "nope", "nada")
    assert not res2.ok


def test_editor_rolls_back_invalid_python(tmp_path):
    f = tmp_path / "app.py"
    Editor.write_file(str(f), "def foo():\n    return 1\n")
    res = Editor.apply_replace(str(f), "def foo():", "def foo(:\n")
    # either rejected or rolled back; file must still be valid
    assert not res.ok
    assert "def foo():" in f.read_text()


def test_diff_block_regex():
    diff = "<<<<<<< SEARCH\nold text\n=======\nnew text\n>>>>>>> REPLACE"
    m = REPLACE_BLOCK.search(diff)
    assert m
    assert m.group("old") == "old text"
    assert m.group("new") == "new text"
