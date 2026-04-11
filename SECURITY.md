# Security Audit — Hardcoded API Keys

> **ACTION REQUIRED**: Before pushing to any remote, remove all hardcoded keys and rotate them.

## In Your Original Codebase (CHECK THESE)

```bash
grep -rn "hf_\|sk-\|AIza\|gsk_\|AKIA" *.py
```

Replace with:
```python
# BEFORE
HF_TOKEN = "hf_abc123..."
# AFTER
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN: raise ValueError("Set HF_TOKEN environment variable")
```

Rotate all keys that were ever committed — they exist in git history.

The `.gitignore` excludes `.env` files. Never commit `.env`.
