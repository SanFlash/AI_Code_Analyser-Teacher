# AI Code Analyzer (Flask)

A web app to paste or upload Python code and get an analysis report: syntax check, lint diagnostics, complexity/maintainability metrics, plus optional AI suggestions and a proposed corrected code.

## What's fixed
- **Security:** removed hardcoded OpenAI key. Use `OPENAI_API_KEY` env var if you want AI suggestions.
- **Reliability:** `/api/run` is now registered before the app starts (route ordering fixed).
- **Structure:** files placed under `/templates` and `/static` to match Flask defaults.
- **Version:** v1.1.0

## Quickstart

```bash
# 1) Create venv
python -m venv .venv

# 2) Activate
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) (Optional) Enable OpenAI-powered suggestions
# PowerShell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL   = "gpt-4o-mini"
# bash/zsh
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"

# 5) Run the app
python app.py
# Open http://localhost:5000
```

## Endpoints
- `GET /` UI
- `POST /api/analyze` JSON: `{ code: string, language: 'auto'|'python' }` → `{ diagnostics, summary, metrics, suggestions, proposed_code }`
- `POST /api/run` JSON: `{ code: string, language: 'auto'|'python' }` → `{ stdout, stderr, exit_code, runtime_ms }`
- `GET /api/health` → environment + package info

## Notes
- If `pylint`, `flake8`, or `radon` are missing, the app still works and returns partial results.
- Executing arbitrary code is risky. Keep `EXEC_TIMEOUT_SEC` small and **do not** expose this to untrusted users.

## Configuration
Environment variables:
- `OPENAI_API_KEY`: Enables LLM-based suggestions when set.
- `OPENAI_MODEL`: Model name (default: `gpt-4o-mini`).
- `EXEC_TIMEOUT_SEC`: Max seconds to allow user code to run (default: `3`, min `0.1`, max `30`).
- `MAX_OUTPUT_CHARS`: Truncate stdout/stderr after this many characters (default: `10000`, min `1000`, max `500000`).
