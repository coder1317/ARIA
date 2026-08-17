---
name: code-review
description: Structured code review across 6 categories
triggers: review, code review, inspect code, check my code
---

When reviewing code, check in this order:
1. Bugs — logic errors, wrong conditions, off-by-one
2. Syntax — will it parse?
3. Missing imports / undefined names
4. API misuse — wrong function signatures
5. Security — injection, unsafe eval, hardcoded secrets
6. Style — but only flag what blocks readability

Report each issue as: `ISSUE: <category> | <file> | <problem>`
If clean, output exactly: `NO ISSUES`
