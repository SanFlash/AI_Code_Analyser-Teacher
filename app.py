import json
import os
import subprocess
import tempfile
import sys
from typing import Dict, List, Tuple, Any

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# Serve templates from ./templates and static assets from ./static
app = Flask(__name__, static_folder="static", template_folder="templates")

load_dotenv()

# NOTE: Hardcoded API keys are a security risk. We do NOT embed any fallback key.
# If you want OpenAI-powered suggestions, set the OPENAI_API_KEY environment variable.

def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
        openai.api_key = api_key
        return openai
    except Exception:
        return None


def detect_language_from_filename(filename: str) -> str:
    if not filename:
        return "auto"
    _, ext = os.path.splitext(filename.lower())
    if ext == ".py":
        return "python"
    if ext in {".js", ".mjs", ".cjs"}:
        return "javascript"
    if ext in {".html", ".htm"}:
        return "html"
    return "auto"


def parse_python_syntax(code: str) -> Tuple[bool, Dict]:
    try:
        import ast  # local import to speed cold start
        ast.parse(code)
        return True, {}
    except SyntaxError as exc:  # pragma: no cover - best effort
        detail = {
            "type": "error",
            "message": exc.msg,
            "line": exc.lineno or 1,
            "column": exc.offset or 0,
            "symbol": "syntax-error",
        }
        return False, detail


def run_black(code: str) -> str:
    # Stub for potential formatting; intentionally no-op by default
    return code


def compute_python_metrics(code: str) -> Dict[str, Any]:
    """Compute complexity and maintainability metrics using Radon if available."""
    metrics: Dict[str, Any] = {
        "cyclomatic": {"average": None, "max": None, "functions": []},
        "maintainability_index": None,
    }
    try:
        from radon.complexity import cc_visit
        from radon.metrics import mi_compute

        blocks = cc_visit(code)  # list of objects with .name, .complexity, .lineno
        if blocks:
            complexities = [getattr(b, "complexity", None) for b in blocks if getattr(b, "complexity", None) is not None]
            if complexities:
                avg = sum(complexities) / max(len(complexities), 1)
                metrics["cyclomatic"]["average"] = round(avg, 2)
                metrics["cyclomatic"]["max"] = max(complexities)
            metrics["cyclomatic"]["functions"] = [
                {
                    "name": getattr(b, "name", "<unknown>"),
                    "complexity": getattr(b, "complexity", None),
                    "line": getattr(b, "lineno", None),
                }
                for b in blocks
            ]

        # Overall file MI (0-100, higher is better)
        try:
            mi_val = mi_compute(code, multi=False)
            metrics["maintainability_index"] = round(float(mi_val), 2)
        except Exception:
            pass
    except Exception:
        # radon not installed or failed; keep defaults
        pass

    return metrics


def run_pylint_on_code(code: str) -> List[Dict]:
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tf:
            tf.write(code)
            temp_file_path = tf.name

        command = [
            sys.executable, "-m", "pylint",
            "--output-format=json", "--score=n", "--reports=n",
            "--disable=import-error",
            temp_file_path,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )

        output = completed.stdout.strip()
        if not output:
            return []

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []

        diagnostics: List[Dict] = []
        for item in data:
            diagnostics.append(
                {
                    "type": (item.get("type") or "warning").lower(),
                    "line": item.get("line") or 1,
                    "column": item.get("column") or 0,
                    "message": item.get("message") or "",
                    "symbol": item.get("symbol") or item.get("message-id") or "",
                }
            )
        return diagnostics
    except Exception:
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def execute_python(code: str, desired_output: str | None = None) -> Dict[str, Any]:
    """Execute Python code in a subprocess and capture stdout/stderr.

    WARNING: This executes arbitrary code. Do not expose to untrusted users in production.
    We apply timeouts and isolation flags, but it is not a secure sandbox.
    """
    import time
    import difflib
    tmp_file = None
    start = time.perf_counter()
    try:
        try:
            timeout_sec = float(os.environ.get("EXEC_TIMEOUT_SEC", "3"))
            if not (0.1 <= timeout_sec <= 30):
                timeout_sec = 3.0
        except Exception:
            timeout_sec = 3.0
        try:
            max_output_chars = int(os.environ.get("MAX_OUTPUT_CHARS", "10000"))
            if max_output_chars < 1000:
                max_output_chars = 1000
            if max_output_chars > 500000:
                max_output_chars = 500000
        except Exception:
            max_output_chars = 10000

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tf:
            tf.write(code)
            tmp_file = tf.name

        cmd = [sys.executable, "-I", "-B", "-S", tmp_file]
        completed = subprocess.run(
            cmd,
            input="",
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        runtime_ms = int((time.perf_counter() - start) * 1000)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        code_rc = completed.returncode

        def _truncate(s: str, max_len: int = 10000) -> tuple[str, bool]:
            if len(s) > max_len:
                return s[:max_len] + "\\n...[truncated]", True
            return s, False

        stdout, out_trunc = _truncate(stdout, max_output_chars)
        stderr, err_trunc = _truncate(stderr, max_output_chars)

        diff_text = None
        if desired_output is not None:
            a_lines = (stdout or "").splitlines(keepends=True)
            b_lines = (desired_output or "").splitlines(keepends=True)
            import difflib as _difflib
            udiff = _difflib.unified_diff(a_lines, b_lines, fromfile="actual", tofile="desired", lineterm="")
            diff_text = "".join(udiff)

        return {
            "success": True,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": code_rc,
            "runtime_ms": runtime_ms,
            "stdout_truncated": out_trunc,
            "stderr_truncated": err_trunc,
            "diff": diff_text,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": True,
            "stdout": "",
            "stderr": "Timed out",
            "exit_code": 124,
            "runtime_ms": int((time.perf_counter() - start) * 1000),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "diff": None,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


def run_flake8_on_code(code: str) -> List[Dict]:
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tf:
            tf.write(code)
            temp_file_path = tf.name

        completed = subprocess.run(
            [sys.executable, "-m", "flake8", "--max-line-length=120", temp_file_path],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )

        stdout = completed.stdout.strip()
        if not stdout:
            return []

        diagnostics: List[Dict] = []
        for line in stdout.splitlines():
            try:
                prefix, message = line.split(": ", 1)
                filename, line_no, col_no, code = prefix.split(":")
                message = f"{code.strip()} {message.strip()}"
                diagnostics.append(
                    {
                        "type": "warning",
                        "line": int(line_no),
                        "column": int(col_no),
                        "message": message,
                        "symbol": code.strip(),
                    }
                )
            except Exception:
                continue

        return diagnostics
    except Exception:
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


def analyze_python(code: str) -> Dict:
    is_syntax_ok, syntax_detail = parse_python_syntax(code)

    diagnostics: List[Dict] = []
    if not is_syntax_ok:
        diagnostics.append(syntax_detail)
    else:
        diagnostics = run_pylint_on_code(code) + run_flake8_on_code(code)

    summary = {"errors": 0, "warnings": 0, "conventions": 0, "refactors": 0}
    for d in diagnostics:
        level = (d.get("type") or "").lower()
        if level.startswith("error") or level in {"error", "fatal"}:
            summary["errors"] += 1
        elif level.startswith("warn") or level == "warning":
            summary["warnings"] += 1
        elif level.startswith("convention") or level == "convention":
            summary["conventions"] += 1
        elif level.startswith("refactor") or level == "refactor":
            summary["refactors"] += 1

    metrics = compute_python_metrics(code)

    return {
        "language": "python",
        "original_code": code,
        "diagnostics": diagnostics,
        "summary": summary,
        "metrics": metrics,
    }


def ai_review_and_fix(code: str, diagnostics: List[Dict], metrics: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str | None]:
    """Use OpenAI to produce suggestions and a proposed corrected code. Returns (suggestions, proposed_code)."""
    client = _get_openai_client()
    if client is None:
        return [], None
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    condensed_diags: List[Dict[str, Any]] = []
    try:
        for d in (diagnostics or [])[:25]:
            condensed_diags.append({
                "type": (d.get("type") or "").lower(),
                "line": d.get("line"),
                "column": d.get("column"),
                "symbol": d.get("symbol"),
                "message": d.get("message"),
            })
    except Exception:
        condensed_diags = []

    system_msg = (
        "You are a senior Python reviewer. Analyze the code and produce precise, actionable suggestions, "
        "then propose a corrected code version that addresses the suggestions while preserving original intent."
    )
    max_code_chars = 40000
    code_snippet = code[:max_code_chars]
    user_msg = (
        "Provide output strictly as JSON.\\n\\n"
        "Context diagnostics (first 25):\\n" + json.dumps(condensed_diags)[:4000] + "\\n\\n"
        "Metrics: " + json.dumps(metrics) + "\\n\\n"
        f"Code (truncated to {max_code_chars} chars if long):\\n" + code_snippet
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=1600,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "analysis_and_fix",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["suggestions", "proposed_code"],
                        "properties": {
                            "suggestions": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["title", "detail"],
                                    "properties": {
                                        "title": {"type": "string"},
                                        "detail": {"type": "string"},
                                        "line": {"type": "integer"},
                                        "category": {"type": "string"}
                                    }
                                }
                            },
                            "proposed_code": {"type": "string"}
                        }
                    }
                }
            },
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        suggestions = data.get("suggestions") if isinstance(data, dict) else []
        proposed_code = data.get("proposed_code") if isinstance(data, dict) else None
        if not isinstance(suggestions, list):
            suggestions = []
        return suggestions, proposed_code
    except Exception:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=1600,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            data = json.loads(content)
            suggestions = data.get("suggestions") if isinstance(data, dict) else []
            proposed_code = data.get("proposed_code") if isinstance(data, dict) else None
            if not isinstance(suggestions, list):
                suggestions = []
            return suggestions, proposed_code
        except Exception:
            return [], None


def _heuristic_suggestions_python(code: str, diagnostics: List[Dict], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for d in diagnostics:
        if (d.get("symbol") or d.get("type") or "").startswith("syntax") or d.get("type") == "error":
            suggestions.append({
                "title": "Fix syntax error",
                "detail": f"There is a syntax error at line {d.get('line')}. Review punctuation, indentation, and unmatched brackets.",
                "line": d.get("line"),
            })
            break

    for d in diagnostics:
        msg = (d.get("message") or "").lower()
        sym = (d.get("symbol") or "").lower()
        if "unused" in msg or sym in {"f401", "w0611", "w0612"}:
            suggestions.append({
                "title": "Remove unused imports/variables",
                "detail": "Use autoflake or manually remove unused imports/variables to reduce lint noise and speed up execution.",
                "line": d.get("line"),
            })
            break

    try:
        cyclo = metrics.get("cyclomatic") or {}
        max_c = cyclo.get("max")
        if max_c is not None and max_c >= 10:
            worst = None
            funcs = cyclo.get("functions") or []
            if funcs:
                worst = max(funcs, key=lambda f: f.get("complexity") or 0)
            fn_label = (worst or {}).get("name") or "function/method"
            line_no = (worst or {}).get("line")
            suggestions.append({
                "title": "Refactor high-complexity function",
                "detail": f"{fn_label} has cyclomatic complexity {max_c}. Extract helpers, reduce branching, and early-return to simplify.",
                "line": line_no,
            })
    except Exception:
        pass

    try:
        mi = metrics.get("maintainability_index")
        if isinstance(mi, (int, float)) and mi < 60:
            suggestions.append({
                "title": "Improve maintainability",
                "detail": "Maintainability Index is low. Add docstrings, reduce code duplication, shorten long functions, and improve naming.",
                "line": None,
            })
    except Exception:
        pass

    for d in diagnostics:
        if (d.get("symbol") or "").upper() in {"E501"} or "line too long" in (d.get("message") or "").lower():
            suggestions.append({
                "title": "Wrap long lines",
                "detail": "Keep lines under 120 chars (configured). Break expressions or use implicit line joins.",
                "line": d.get("line"),
            })
            break
    return suggestions


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    def _ver(pkg: str):
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            return None
    info = {
        "status": "ok",
        "python": sys.version.split(" ")[0],
        "packages": {
            "pylint": _ver("pylint"),
            "flake8": _ver("flake8"),
            "radon": _ver("radon"),
            "openai": _ver("openai"),
        },
        "openai": {
            "enabled": bool(os.environ.get("OPENAI_API_KEY")),
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        },
        "execution_limits": {
            "timeout_sec": os.environ.get("EXEC_TIMEOUT_SEC", "3"),
            "max_output_chars": os.environ.get("MAX_OUTPUT_CHARS", "10000"),
        },
        "version": "1.1.0",
    }
    return jsonify(info)


@app.errorhandler(404)
def handle_404(err):
    try:
        path = request.path or ""
    except Exception:
        path = ""
    if path.startswith("/api/"):
        return jsonify({"success": False, "error": "Not found", "status": 404}), 404
    return err, 404


@app.errorhandler(405)
def handle_405(err):
    try:
        path = request.path or ""
    except Exception:
        path = ""
    if path.startswith("/api/"):
        return jsonify({"success": False, "error": "Method not allowed", "status": 405}), 405
    return err, 405


@app.errorhandler(500)
def handle_500(err):
    try:
        path = request.path or ""
    except Exception:
        path = ""
    if path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal server error", "status": 500}), 500
    return err, 500


@app.post("/api/analyze")
def analyze():
    code: str = ""
    language: str = "auto"

    payload = request.get_json(silent=True)
    if payload:
        code = (payload.get("code") or "").rstrip("\\n")
        language = (payload.get("language") or "auto").lower()
    else:
        if "file" in request.files:
            f = request.files["file"]
            language = detect_language_from_filename(f.filename)
            code = f.read().decode("utf-8", errors="replace")
        else:
            code = (request.form.get("code") or "").rstrip("\\n")
            language = (request.form.get("language") or "auto").lower()

    if not code:
        return jsonify({"success": False, "error": "No code provided"}), 400

    if language in {"auto", "python"}:
        result = analyze_python(code)
        try:
            suggestions, proposed_code = ai_review_and_fix(
                result.get("original_code") or code,
                result.get("diagnostics") or [],
                result.get("metrics") or {},
            )
        except Exception:
            suggestions, proposed_code = [], None
        if not suggestions:
            suggestions = _heuristic_suggestions_python(result.get("original_code") or code, result.get("diagnostics") or [], result.get("metrics") or {})
        return jsonify({"success": True, **result, "suggestions": suggestions, "proposed_code": proposed_code})

    return jsonify(
        {
            "success": True,
            "language": language,
            "original_code": code,
            "diagnostics": [
                {
                    "type": "info",
                    "line": 1,
                    "column": 0,
                    "message": f"Language '{language}' not yet supported. Python is supported.",
                    "symbol": "not-supported",
                }
            ],
            "summary": {"errors": 0, "warnings": 0, "conventions": 0, "refactors": 0},
        }
    )


def explain_code(code: str) -> str:
    """Return a simple, beginner-friendly explanation of the code in plain English (no code blocks or technical details)."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Code explanation is unavailable (Google API key not set)."
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Explain the following Python code to a complete beginner. "
            "Use only simple English sentences. "
            "Do not include any code blocks, technical terms, or step-by-step operations. "
            "Just describe what the code does in plain language, as if you are talking to someone with no programming experience.\n\n"
            f"{code}\n"
        )
        response = model.generate_content(prompt)
        return getattr(response, "text", str(response)).strip()
    except Exception as e:
        return f"Code explanation could not be generated due to an error: {str(e)}"


@app.post("/api/run")
def run_code():
    payload = request.get_json(silent=True) or {}
    code = (payload.get("code") or "").rstrip("\n")
    language = (payload.get("language") or "auto").lower()
    desired_output = payload.get("desired_output")
    if not code:
        return jsonify({"success": False, "error": "No code provided"}), 400
    if language not in {"auto", "python"}:
        return jsonify({"success": False, "error": f"Language '{language}' not supported for execution"}), 400
    result = execute_python(code, desired_output)
    explanation = explain_code(code)
    return jsonify({"success": True, **result, "explanation": explanation})


if __name__ == "__main__":
    port_str = os.environ.get("PORT", "5000")
    try:
        port = int(port_str)
    except ValueError:
        port = 5000
    app.run(host="0.0.0.0", port=port, debug=True)
