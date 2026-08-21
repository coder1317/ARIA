---
description: Security audit of code, dependencies, and configurations
triggers: security, audit, vulnerability, scan, penetration
---

# Security Code Audit Skill

## Checklist
1. **Input Validation** — all user inputs sanitized
2. **SQL Injection** — parameterized queries only
3. **XSS** — output encoding, CSP headers
4. **Auth** — JWT expiry, refresh tokens, bcrypt passwords
5. **Secrets** — no hardcoded keys, use env vars
6. **Dependencies** — check `pip-audit`, `npm audit`
7. **File Access** — path traversal prevention
8. **Rate Limiting** — API endpoints protected
9. **CORS** — restrictive origin policy
10. **Logging** — no secrets in logs

## Python Security Tools
```bash
pip-audit                    # Check Python dependencies
bandit -r .                  # Static security analysis
safety check                 # Known vulnerabilities
semgrep --config auto .      # Pattern-based scanning
```

## Common Vulnerabilities to Check
- `eval()` / `exec()` on user input
- `os.system()` / `subprocess(shell=True)`
- `pickle.loads()` on untrusted data
- `yaml.load()` without `SafeLoader`
- `assert` used for validation (stripped in -O)
- Timing attacks on secrets comparison
