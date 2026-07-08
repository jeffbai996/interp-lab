#!/usr/bin/env python3
"""Build a solved copy of an exercise notebook (TODO cells replaced by their
paired <details>Solution</details> code) and execute it end to end.

Notebooks 01-04 are exercises: a code cell with `...`/TODO placeholders is
followed by a markdown cell holding the worked solution in a fenced python
block. The committed notebook is not meant to run as-is; this script produces
the notebook a reader ends up with after solving each exercise, so CI can
verify *that* still executes against current library versions.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

SOLUTION_RE = re.compile(r"```python\n(.*?)\n```", re.DOTALL)


BARE_ELLIPSIS_RE = re.compile(r"^\s*\.\.\.\s*$")
ASSIGN_ELLIPSIS_RE = re.compile(r"^\s*\S.*=\s*\.\.\.\s*$")


def is_marker_line(line: str) -> bool:
    return "TODO" in line or bool(BARE_ELLIPSIS_RE.match(line)) or bool(ASSIGN_ELLIPSIS_RE.match(line))


def has_assign_placeholder(code: str) -> bool:
    """`name = ...` markers (vs a bare `...` body) signal the solution
    redefines named values, possibly reordered relative to the cell's other
    setup lines — not safe to splice in place, replace the whole cell."""
    return any(ASSIGN_ELLIPSIS_RE.match(line) for line in code.split("\n"))


def has_placeholder(code: str) -> bool:
    return any(is_marker_line(line) for line in code.split("\n"))


def extract_solution(markdown_src: str) -> str | None:
    if "Solution" not in markdown_src or "<details>" not in markdown_src:
        return None
    m = SOLUTION_RE.search(markdown_src)
    return m.group(1) if m else None


def solve_notebook(path: Path) -> nbformat.NotebookNode:
    """Replace a placeholder cell's body with the paired solution's code,
    keeping any trailing non-placeholder lines (typically the payoff
    `print(...)`/plot calls that come after the TODO section and aren't
    themselves redefined in the solution block)."""
    nb = nbformat.read(path, as_version=4)
    cells = nb["cells"]
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        if cell["cell_type"] != "code" or not has_placeholder(src):
            continue
        if i + 1 >= len(cells) or cells[i + 1]["cell_type"] != "markdown":
            raise ValueError(f"{path.name} cell {i}: placeholder with no following solution cell")
        solution = extract_solution("".join(cells[i + 1]["source"]))
        if solution is None:
            raise ValueError(f"{path.name} cell {i}: no fenced python solution found")

        lines = src.split("\n")
        marker_idxs = [j for j, line in enumerate(lines) if is_marker_line(line)]
        first_marker, last_marker = min(marker_idxs), max(marker_idxs)

        if has_assign_placeholder(src):
            leading = "\n".join(lines[:first_marker]).strip("\n")
            trailing = "\n".join(lines[last_marker + 1 :]).strip("\n")
            parts = [p for p in (leading, solution, trailing) if p]
            cell["source"] = "\n\n".join(parts)
        else:
            kept = [line for j, line in enumerate(lines) if j not in marker_idxs]
            insert_at = sum(1 for j in range(last_marker + 1) if j not in marker_idxs)
            kept[insert_at:insert_at] = [solution]
            cell["source"] = "\n".join(kept).strip("\n")
        cell["outputs"] = []
        cell["execution_count"] = None
    return nb


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: solve_and_execute.py <notebook.ipynb> <output_dir>", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    solved = solve_notebook(src)
    solved_path = out_dir / f"solved_{src.name}"
    nbformat.write(solved, solved_path)
    print(f"wrote {solved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
