"""Tests for ultra.evaluator — project scoring + circuit breaker."""
import time

from ultra.evaluator import CircuitBreaker, Evaluator


def test_good_project_scores_well(tmp_path):
    (tmp_path / "app.py").write_text("def main():\n    print('hi')\n\nif __name__ == '__main__':\n    main()\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    result = Evaluator().evaluate_project(tmp_path)
    assert result.passed
    assert result.score >= 0.6
    assert result.checks["correctness"]["ok"]


def test_syntax_error_lowers_score(tmp_path):
    (tmp_path / "app.py").write_text("def broken(:\n    print('x')\n")
    (tmp_path / "ok.py").write_text("x = 1\n")
    result = Evaluator().evaluate_project(tmp_path)
    assert not result.checks["correctness"]["ok"]
    assert result.checks["correctness"]["score"] == 0.5


def test_secret_in_code_fails_safety(tmp_path):
    (tmp_path / "app.py").write_text("key = 'sk-proj-abcdefghijklmnopqrstuvwxyz123'\n")
    result = Evaluator().evaluate_project(tmp_path)
    assert not result.checks["safety"]["ok"]


def test_empty_project_fails(tmp_path):
    result = Evaluator().evaluate_project(tmp_path)
    assert not result.passed


def test_smoke_runs_working_entry_point(tmp_path):
    (tmp_path / "main.py").write_text(
        "import sys\n"
        "def main():\n"
        "    print('hello')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    result = Evaluator().evaluate_project(tmp_path)
    assert result.checks["smoke"]["ok"]
    assert "ran cleanly" in result.checks["smoke"]["detail"]


def test_smoke_fails_when_entry_point_crashes(tmp_path):
    (tmp_path / "main.py").write_text(
        "import sys\n"
        "def main():\n"
        "    raise RuntimeError('boom')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    result = Evaluator().evaluate_project(tmp_path)
    assert not result.checks["smoke"]["ok"]
    assert "exited" in result.checks["smoke"]["detail"]


def test_smoke_skipped_when_security_critical(tmp_path):
    (tmp_path / "main.py").write_text(
        "import sys\n"
        "x = eval('1+1')\n"
        "def main():\n"
        "    print('hi')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    result = Evaluator().evaluate_project(tmp_path)
    assert not result.checks["safety"]["ok"]
    assert "skipped" in result.checks["smoke"]["detail"]


def test_no_entry_point_detected(tmp_path):
    (tmp_path / "lib.py").write_text("def helper():\n    return 1\n")
    result = Evaluator().evaluate_project(tmp_path)
    assert not result.checks["smoke"]["ok"]
    assert "no entry point" in result.checks["smoke"]["detail"]


def test_circuit_breaker_opens_and_recovers():
    cb = CircuitBreaker("build", max_failures=3, cooldown_sec=0.05)
    assert not cb.is_open
    assert cb.record_failure() is False
    assert cb.record_failure() is False
    assert cb.record_failure() is True  # tripped
    assert cb.is_open
    assert cb.remaining() > 0
    time.sleep(0.1)
    assert not cb.is_open  # cooled down
    assert cb.failures == 0


def test_circuit_success_resets():
    cb = CircuitBreaker("build", max_failures=3)
    cb.record_failure()
    cb.record_success()
    assert cb.failures == 0
    assert not cb.is_open
