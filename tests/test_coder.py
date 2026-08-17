import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.agents.coder import _parse_files


def test_marker_format():
    raw = """---FILE: app.py ---
print("hi")
---FILE: README.md ---
# app
"""
    files = _parse_files(raw)
    assert files["app.py"].strip() == 'print("hi")'
    assert files["README.md"].strip() == "# app"


def test_fence_format_with_filename():
    raw = """Here is the code:
```python calculator.py
def add(a, b):
    return a + b
```
And the readme:
```markdown README.md
# Calculator
```
"""
    files = _parse_files(raw)
    assert "def add" in files.get("calculator.py", "")
    assert "# Calculator" in files.get("README.md", "")


def test_fence_without_filename_skipped():
    raw = "```python\ndef foo(): pass\n```"
    assert _parse_files(raw) == {}


def test_marker_with_fence_inside():
    raw = """---FILE: app.py ---
```python
print("ok")
```
"""
    files = _parse_files(raw)
    assert "print" in files.get("app.py", "")
