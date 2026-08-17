import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultra.config import Config, _load_dotenv


def test_defaults():
    cfg = Config.load()
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.chat_model  # non-empty
    assert cfg.data_dir  # non-empty


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_CHAT_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text("OLLAMA_CHAT_MODEL=llama3.2:3b\n")
    _load_dotenv(env)
    assert os.getenv("OLLAMA_CHAT_MODEL") == "llama3.2:3b"


def test_ensure_dirs(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", projects_dir=tmp_path / "proj")
    cfg.ensure_dirs()
    assert (tmp_path / "data" / "memory").is_dir()
