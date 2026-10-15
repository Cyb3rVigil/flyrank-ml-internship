#!/usr/bin/env python3
"""Static validation for the FlyRank internship notebooks.

This checks notebook JSON/structure and parses Python code cells after removing
IPython-only magics and shell commands. It intentionally does not access the
gated warehouse or execute data/model cells; run those in Colab with HF_TOKEN.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = (ROOT / "notebooks", ROOT / "work" / "notebooks")
BAD_SECRET_RE = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+SECRET\s*\(\s*TYPE\s+HUGGINGFACE",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
errors: list[str] = []
checked_notebooks = 0
checked_code_cells = 0
skipped_magic_cells = 0

for directory in NOTEBOOK_DIRS:
    if not directory.exists():
        errors.append(f"Required notebook directory is missing: {directory.relative_to(ROOT)}")
        continue

    for path in sorted(directory.glob("*.ipynb")):
        checked_notebooks += 1
        relative = path.relative_to(ROOT)
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{relative}: invalid notebook JSON: {exc}")
            continue

        if notebook.get("nbformat") != 4:
            errors.append(f"{relative}: expected nbformat 4, got {notebook.get('nbformat')!r}")
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            errors.append(f"{relative}: 'cells' must be a list")
            continue

        # Warehouse notebooks must explicitly load DuckDB's HTTPFS extension
        # before using hf:// URLs or Hugging Face secrets.
        all_code = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else str(cell.get("source", ""))
            for cell in cells
            if cell.get("cell_type") == "code"
        )
        if (
            "hf://datasets/FlyRank/internship-warehouse" in all_code
            and "CREATE OR REPLACE SECRET hf" in all_code
            and not re.search(r"\bLOAD\s+httpfs\b", all_code, re.IGNORECASE)
        ):
            errors.append(
                f"{relative}: warehouse notebook must LOAD httpfs before accessing hf://"
            )

        for cell_index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                errors.append(f"{relative} cell {cell_index}: cell must be an object")
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not isinstance(source, str):
                errors.append(f"{relative} cell {cell_index}: source must be text or a text list")
                continue
            if TOKEN_RE.search(source):
                errors.append(f"{relative} cell {cell_index}: possible hard-coded Hugging Face token")
            if BAD_SECRET_RE.search(source):
                errors.append(
                    f"{relative} cell {cell_index}: malformed Hugging Face secret; "
                    "name the secret, e.g. CREATE OR REPLACE SECRET hf (TYPE huggingface, ...)"
                )

            if cell.get("cell_type") != "code":
                continue
            checked_code_cells += 1
            lines = source.splitlines()
            first_nonempty = next((line.strip() for line in lines if line.strip()), "")
            # Cell magics change the entire cell language (%%bash, %%capture, etc.).
            if first_nonempty.startswith("%%"):
                skipped_magic_cells += 1
                continue

            sanitized = []
            for line in lines:
                stripped = line.lstrip()
                # Remove IPython line and shell magics; preserve a statement so
                # constructs that contain only a magic are syntactically valid.
                if stripped.startswith("%") or stripped.startswith("!"):
                    sanitized.append("pass  # stripped IPython magic for syntax validation")
                else:
                    sanitized.append(line)
            python_source = "\n".join(sanitized)
            try:
                ast.parse(python_source, filename=f"{relative}:cell-{cell_index}")
            except SyntaxError as exc:
                errors.append(
                    f"{relative} cell {cell_index}: Python syntax error at line "
                    f"{exc.lineno}: {exc.msg}"
                )

paper_url = ROOT / "submission" / "paper_url.txt"
if not paper_url.exists():
    errors.append("Missing required submission/paper_url.txt")
else:
    value = paper_url.read_text(encoding="utf-8")
    lines = value.splitlines()
    if len(lines) != 1 or not lines[0].startswith("https://") or not lines[0].strip():
        errors.append("submission/paper_url.txt must contain exactly one https:// URL on one line")

print(f"Notebook files checked: {checked_notebooks}")
print(f"Code cells syntax-checked: {checked_code_cells}")
print(f"Magic-based cells skipped: {skipped_magic_cells}")
if errors:
    print(f"FAIL: {len(errors)} issue(s) found:")
    for error in errors:
        print(f" - {error}")
    sys.exit(1)
print("PASS: notebook JSON, code-cell syntax, data-secret guard, and paper URL receipt.")
