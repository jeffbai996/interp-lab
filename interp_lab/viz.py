"""Zero-dependency heatmap — a colored HTML table, no plotting libraries.

A 12×N grid of "% of behavior restored" (or attention weights, or DLA) is the
workhorse visual of this lab, and a colored table renders it perfectly inside a
notebook with nothing to install. Red = positive, blue = negative, white = zero.
"""
from __future__ import annotations

from IPython.display import HTML, display


def heatmap(grid, col_labels, row_labels, title=""):
    """grid: anything indexable as [row][col] of values in roughly −0.35 … 1.05.
    Cells show value × 100; the color scale is tuned for the IOI metric range."""

    def color(v):
        x = max(min(float(v), 1.05), -0.35)
        if x >= 0:
            t = x / 1.05
            return f"rgb(255,{int(255 * (1 - 0.75 * t))},{int(255 * (1 - 0.85 * t))})"
        t = x / -0.35
        return f"rgb({int(255 * (1 - 0.75 * t))},{int(255 * (1 - 0.5 * t))},255)"

    th = "padding:2px 7px;font-size:11px"
    head = "<tr><th></th>" + "".join(f"<th style='{th}'>{c}</th>" for c in col_labels) + "</tr>"
    body = "".join(
        f"<tr><th style='{th};text-align:right'>{rl}</th>"
        + "".join(
            f"<td style='background:{color(v)};{th};text-align:right'>{float(v) * 100:.0f}</td>"
            for v in row
        )
        + "</tr>"
        for rl, row in zip(row_labels, grid)
    )
    display(
        HTML(
            f"<div style='font-family:monospace'><b>{title}</b>"
            f"<table style='border-collapse:collapse'>{head}{body}</table></div>"
        )
    )
