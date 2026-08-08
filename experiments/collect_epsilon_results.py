#!/usr/bin/env python3
"""Combine B2 delta directories and deduplicate identical objective points."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    from .collect_main_results import PER_RUN_COLS, SUMMARY_COLS
except ImportError:
    from collect_main_results import PER_RUN_COLS, SUMMARY_COLS


UNIQUE_COLUMNS = [
    "instance_name",
    "cfg_id",
    "label",
    "cardinality",
    "ic",
    "sb",
    "coverage",
    "similarity",
    "continuity",
    "overtime",
    "similarity_reference_optimum",
    "similarity_realized_loss_absolute",
    "delta_count",
    "deltas",
    "minimum_delta",
    "maximum_delta",
    "mean_elapsed_s",
    "pareto_nondominated",
    "dominated_by_points",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def _write_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            extrasaction="ignore",
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, TypeError):
        return Decimal("Infinity")


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: str) -> int:
    try:
        decimal = Decimal(value)
    except (InvalidOperation, TypeError):
        return 0
    if decimal != decimal.to_integral_value():
        raise ValueError(f"expected an integral metric, received {value!r}")
    return int(decimal)


def deduplicate_points(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "OPTIMUM":
            continue
        key = tuple(
            row.get(column, "")
            for column in (
                "instance_name",
                "cfg_id",
                "label",
                "cardinality",
                "ic",
                "sb",
                "coverage",
                "similarity",
                "continuity",
                "overtime",
                "similarity_reference_optimum",
            )
        )
        groups[key].append(row)

    unique_rows: list[dict[str, object]] = []
    for key, group in groups.items():
        deltas = sorted(
            {row.get("delta", "") for row in group},
            key=_decimal,
        )
        elapsed = [
            value
            for row in group
            if (value := _float(row.get("elapsed_s", ""))) is not None
        ]
        reference = _integer(key[10])
        similarity = _integer(key[7])
        unique_rows.append(
            {
                "instance_name": key[0],
                "cfg_id": key[1],
                "label": key[2],
                "cardinality": key[3],
                "ic": key[4],
                "sb": key[5],
                "coverage": key[6],
                "similarity": key[7],
                "continuity": key[8],
                "overtime": key[9],
                "similarity_reference_optimum": key[10],
                "similarity_realized_loss_absolute": reference - similarity,
                "delta_count": len(deltas),
                "deltas": " | ".join(deltas),
                "minimum_delta": deltas[0],
                "maximum_delta": deltas[-1],
                "mean_elapsed_s": (
                    round(statistics.fmean(elapsed), 6) if elapsed else ""
                ),
            }
        )

    ordered = sorted(
        unique_rows,
        key=lambda row: (
            str(row["instance_name"]),
            int(str(row["cfg_id"])),
            _decimal(str(row["minimum_delta"])),
        ),
    )
    return annotate_pareto(ordered)


def _dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    """Return whether left weakly improves all HCORAP metrics and one strictly."""

    left_values = (
        _integer(str(left["coverage"])),
        _integer(str(left["similarity"])),
        _integer(str(left["continuity"])),
        _integer(str(left["overtime"])),
    )
    right_values = (
        _integer(str(right["coverage"])),
        _integer(str(right["similarity"])),
        _integer(str(right["continuity"])),
        _integer(str(right["overtime"])),
    )
    weak = (
        left_values[0] >= right_values[0]
        and left_values[1] >= right_values[1]
        and left_values[2] <= right_values[2]
        and left_values[3] <= right_values[3]
    )
    return weak and left_values != right_values


def annotate_pareto(points: list[dict[str, object]]) -> list[dict[str, object]]:
    """Annotate dominance only within the same instance and formulation."""

    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for point in points:
        key = tuple(
            str(point.get(column, ""))
            for column in (
                "instance_name",
                "cfg_id",
                "cardinality",
                "ic",
                "sb",
            )
        )
        groups[key].append(point)
    for group in groups.values():
        for point in group:
            dominators = [
                other
                for other in group
                if other is not point and _dominates(other, point)
            ]
            point["pareto_nondominated"] = not dominators
            point["dominated_by_points"] = " | ".join(
                f"SIM={other['similarity']},CONT={other['continuity']},OT={other['overtime']}"
                for other in dominators
            )
    return points


def collect_epsilon_results(
    result_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, object]]]:
    raw_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for delta_dir in sorted(result_root.glob("delta_*")):
        raw_path = delta_dir / "results_per_instance.csv"
        summary_path = delta_dir / "summary_by_config.csv"
        if raw_path.is_file():
            raw_rows.extend(_read_csv(raw_path))
        if summary_path.is_file():
            summary_rows.extend(_read_csv(summary_path))

    raw_rows.sort(
        key=lambda row: (
            row.get("instance_name", ""),
            int(row.get("cfg_id", "0") or 0),
            _decimal(row.get("delta", "")),
        )
    )
    summary_rows.sort(
        key=lambda row: (
            _decimal(row.get("delta", "")),
            int(row.get("cfg_id", "0") or 0),
        )
    )
    unique_rows = deduplicate_points(raw_rows)
    return raw_rows, summary_rows, unique_rows


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} RESULT_ROOT", file=sys.stderr)
        return 2
    result_root = Path(sys.argv[1])
    if not result_root.is_dir():
        print(f"ERROR: not a result directory: {result_root}", file=sys.stderr)
        return 2

    raw_rows, summary_rows, unique_rows = collect_epsilon_results(result_root)
    if not raw_rows:
        print(f"ERROR: no B2 per-delta CSV files found under {result_root}", file=sys.stderr)
        return 2

    _write_csv(
        result_root / "epsilon_results_all_deltas.csv",
        PER_RUN_COLS,
        raw_rows,
    )
    _write_csv(
        result_root / "epsilon_summary_by_delta_config.csv",
        SUMMARY_COLS,
        summary_rows,
    )
    _write_csv(
        result_root / "epsilon_unique_points.csv",
        UNIQUE_COLUMNS,
        unique_rows,
    )
    _write_csv(
        result_root / "epsilon_pareto_frontier.csv",
        UNIQUE_COLUMNS,
        [row for row in unique_rows if row["pareto_nondominated"]],
    )
    print(f"All B2 runs: {len(raw_rows)}")
    print(f"Unique B2 points: {len(unique_rows)}")
    print(
        "Nondominated B2 points: "
        f"{sum(bool(row['pareto_nondominated']) for row in unique_rows)}"
    )
    print(f"Excel-ready B2 tables: {result_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
