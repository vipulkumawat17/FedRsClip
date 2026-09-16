from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# CURRENT EXPERIMENT MODE
# ----------------------
# Epoch-wise visual dissertation reports are intentionally disabled.
# Training and standalone zero-shot evaluation now save numerical CSV results
# and checkpoints only. The old visual-report implementation is not required
# for the current multi-dataset experiments.


def write_csv(
    output_csv: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write tabular experiment results to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        output_csv.write_text("", encoding="utf-8")
        return

    names = fieldnames or list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)