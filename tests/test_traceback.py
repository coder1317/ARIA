import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.agents.engineering import _issues_from_traceback, _quality_gate


def test_quality_gate_accepts_real_project(tmp_path):
    (tmp_path / "app.py").write_text("def main():\n    print('hi')\n\nif __name__ == '__main__':\n    main()\n")
    (tmp_path / "README.md").write_text("# app")
    ok, why = _quality_gate(tmp_path)
    assert ok, why


def test_quality_gate_rejects_empty(tmp_path):
    (tmp_path / "app.py").write_text("")
    ok, why = _quality_gate(tmp_path)
    assert not ok


def test_quality_gate_rejects_weird_files(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / "calculator").write_text("garbage")
    ok, why = _quality_gate(tmp_path)
    assert not ok


def test_parses_project_files(tmp_path):
    (tmp_path / "divide.py").write_text("")
    (tmp_path / "test_calc.py").write_text("")
    output = (
        f'File "{tmp_path}/test_calc.py", line 2, in <module>\n'
        '  from divide import divide\n'
        f'File "{tmp_path}/divide.py", line 1, in <module>\n'
        "    import ValueError\n"
        "ModuleNotFoundError: No module named 'ValueError'\n"
    )
    issues = _issues_from_traceback(output, tmp_path)
    files = {i["file"] for i in issues}
    assert "divide.py" in files
    assert "test_calc.py" in files


def test_external_files_ignored(tmp_path):
    output = 'File "/usr/lib/python3.12/loader.py", line 394\n'
    issues = _issues_from_traceback(output, tmp_path)
    # only the external file → falls back to whole-project repair
    assert issues[0]["file"] == "(project)"
