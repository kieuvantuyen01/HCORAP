#!/usr/bin/env python3
"""
collect_main_results.py
Tổng hợp kết quả một objective policy trên 8 cấu hình thành:
  - results_per_instance.csv  (đã có từ run script, bổ sung nếu cần)
  - summary_by_config.csv     (1 row/config: mean/median/PAR-2 v.v.)
  - summary_by_instance.csv   (so sánh 8 cfg trên cùng instance)

Usage:
    python3 experiments/collect_main_results.py RESULT_DIR
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

# ---------------------------------------------------------------------------
# Cột CSV đầu ra
# ---------------------------------------------------------------------------
PER_RUN_COLS = [
    "cfg_id", "label", "instance_name",
    "method", "objective_mode", "objective_policy",
    "delta", "similarity_reference_optimum", "similarity_lower_bound",
    "similarity_realized_loss_absolute",
    "similarity_realized_loss_fraction", "similarity_realized_loss_ratio",
    "cardinality", "ic", "sb",
    "status",
    "elapsed_s", "timeout_seconds", "solve_s", "solver_calls",
    "variables", "hard_clauses", "soft_clauses",
    "best_value", "weighted_reference_score",
    "similarity", "continuity", "overtime", "coverage",
    "stage_objectives", "stage_optima",
]

SUMMARY_COLS = [
    "cfg_id", "label", "method", "objective_mode", "objective_policy", "delta",
    "cardinality", "ic", "sb",
    "total_runs",
    "optimum_n", "timeout_n", "unsat_n", "error_n",
    "optimum_pct",
    "mean_elapsed_s", "median_elapsed_s", "par2_s",
    "mean_solve_s", "mean_solver_calls",
    "mean_variables", "mean_hard_clauses", "mean_soft_clauses",
    "mean_similarity_reference", "mean_similarity_lower_bound",
    "mean_similarity_realized_loss", "mean_similarity_realized_loss_ratio",
    "mean_best_value", "mean_similarity", "mean_continuity", "mean_overtime",
    "std_best_value",
]

TIMEOUT_SENTINEL = 3600.0  # giây — dùng cho PAR-2


def safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_json_file(path: Path) -> dict:
    """Parse một file JSON kết quả và trả về dict chuẩn hóa."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception as e:
        return {
            "method": "?",
            "objective_mode": "?",
            "objective_policy": "?",
            "delta": "?",
            "similarity_reference_optimum": None,
            "similarity_lower_bound": None,
            "similarity_realized_loss_absolute": None,
            "similarity_realized_loss_fraction": None,
            "similarity_realized_loss_ratio": None,
            "status": "PARSE_ERROR",
            "elapsed_s": 0.0,
            "timeout_seconds": TIMEOUT_SENTINEL,
            "solve_s": 0.0,
            "solver_calls": 0,
            "variables": 0,
            "hard_clauses": 0,
            "soft_clauses": 0,
            "best_value": None,
            "weighted_reference_score": None,
            "similarity": None,
            "continuity": None,
            "overtime": None,
            "coverage": None,
            "stage_objectives": "",
            "stage_optima": "",
            "error": str(e),
        }

    status = d.get("status", "ERROR")
    elapsed = safe_float(d.get("elapsed_seconds"), 0.0)
    timeout_seconds = safe_float(
        d.get("timeout_seconds"), TIMEOUT_SENTINEL
    )

    stages = d.get("stages") or []
    solve_s = sum(s.get("solve_seconds", 0) for s in stages)
    solver_calls = safe_int(d.get("solver_calls"), len(stages))
    variables = max((s.get("variables", 0) for s in stages), default=0)
    hard_cl = max((s.get("hard_clauses", 0) for s in stages), default=0)
    soft_cl = max((s.get("soft_clauses", 0) for s in stages), default=0)
    stage_objectives = " | ".join(
        str(s.get("objective", "")) for s in stages
    )
    stage_optima = " | ".join(str(s.get("optimum", "")) for s in stages)

    m = d.get("metrics") or {}
    sim  = safe_float(m.get("similarity"))
    cont = safe_float(m.get("continuity"))
    ot   = safe_float(m.get("overtime"))
    cov  = safe_float(m.get("coverage"))

    method = d.get("method", "?")
    inferred_mode = {
        "weighted": "weighted",
        "lex-continuity": "lexicographic",
        "lex-overtime": "lexicographic",
        "epsilon": "epsilon-constraint",
    }.get(method, "?")
    inferred_policy = {
        "weighted": "weighted-sum",
        "lex-continuity": "continuity-priority",
        "lex-overtime": "overtime-priority",
        "epsilon": "similarity-budget",
    }.get(method, "?")
    delta = d.get("delta", "") if method == "epsilon" else "-"
    similarity_reference = safe_int(d.get("similarity_reference_optimum"))
    similarity_lower_bound = safe_int(d.get("similarity_lower_bound"))
    realized_loss = safe_int(d.get("similarity_realized_loss_absolute"))
    realized_loss_fraction = d.get("similarity_realized_loss_fraction")
    if similarity_reference is not None and realized_loss is not None:
        realized_loss_ratio = (
            realized_loss / similarity_reference
            if similarity_reference > 0
            else 0.0
        )
    else:
        realized_loss_ratio = None

    wc = safe_float(d.get("continuity_weight"), 1.0)
    wo = safe_float(d.get("overtime_weight"),   1.0)
    penalty = safe_float(
        d.get("overtime_penalty_per_hour"),
        abs(safe_float(d.get("P_value"), -1.0)),
    )
    overtime_cost = safe_float(m.get("overtime_cost"))
    if ot not in (None, 0) and overtime_cost is not None:
        penalty = overtime_cost / ot

    weighted_reference = safe_float(m.get("weighted_reference_score"))
    if (
        weighted_reference is None
        and sim is not None
        and cont is not None
        and ot is not None
    ):
        weighted_reference = sim - wc * cont - wo * penalty * ot

    return {
        "method": method,
        "objective_mode": d.get("objective_mode", inferred_mode),
        "objective_policy": d.get("objective_policy", inferred_policy),
        "delta": delta,
        "similarity_reference_optimum": similarity_reference,
        "similarity_lower_bound": similarity_lower_bound,
        "similarity_realized_loss_absolute": realized_loss,
        "similarity_realized_loss_fraction": realized_loss_fraction,
        "similarity_realized_loss_ratio": realized_loss_ratio,
        "status":       status,
        "elapsed_s":    elapsed,
        "timeout_seconds": timeout_seconds,
        "solve_s":      solve_s,
        "solver_calls": solver_calls,
        "variables":    variables,
        "hard_clauses": hard_cl,
        "soft_clauses": soft_cl,
        # Kept for compatibility with the original B0 CSV schema. For B1/B2
        # this is only a reference score; the policy result is stage_optima.
        "best_value":   weighted_reference,
        "weighted_reference_score": weighted_reference,
        "similarity":   sim,
        "continuity":   cont,
        "overtime":     ot,
        "coverage":     cov,
        "stage_objectives": stage_objectives,
        "stage_optima": stage_optima,
    }


def collect_results(result_dir: Path) -> list[dict]:
    """Duyệt tất cả cfg*/ subdirectory và parse JSON."""
    rows = []
    for cfg_dir in sorted(result_dir.iterdir()):
        if not cfg_dir.is_dir() or not cfg_dir.name.startswith("cfg"):
            continue
        # tên thư mục: cfg1_ORIGINAL, cfg2_SN-none-ss, v.v.
        parts = cfg_dir.name.split("_", 1)
        cfg_id = parts[0].replace("cfg", "")
        label  = parts[1] if len(parts) > 1 else "?"

        for jfile in sorted(cfg_dir.glob("*.json")):
            iname = jfile.stem
            parsed = parse_json_file(jfile)

            # Đọc metadata từ JSON (cardinality, ic, sb)
            try:
                with open(jfile) as fh:
                    d = json.load(fh)
                card = d.get("cardinality_encoding", "?")
                ic   = d.get("implied_constraints", "?")
                sb   = d.get("symmetry_breaking", "?")
            except Exception:
                card = ic = sb = "?"

            row = {
                "cfg_id":        cfg_id,
                "label":         label,
                "instance_name": iname,
                "cardinality":   card,
                "ic":            ic,
                "sb":            sb,
            }
            row.update(parsed)
            rows.append(row)
    return rows


def write_per_instance_csv(rows: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_RUN_COLS,
                           extrasaction="ignore", delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {out_path}  ({len(rows)} rows)")


def compute_par2(rows: list[dict]) -> float:
    """PAR-2 with each run's own timeout; certified UNSAT is successful."""
    success_statuses = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
    vals = []
    for row in rows:
        elapsed = safe_float(row.get("elapsed_s"), 0.0)
        timeout = safe_float(
            row.get("timeout_seconds"), TIMEOUT_SENTINEL
        )
        if row.get("status") in success_statuses:
            vals.append(elapsed)
        else:
            vals.append(2 * timeout)
    return mean(vals) if vals else float("nan")


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    # Never aggregate different objective policies into the same row.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (
            r["cfg_id"],
            r["label"],
            r["method"],
            r["objective_mode"],
            r["objective_policy"],
            r["delta"],
            r["cardinality"],
            r["ic"],
            r["sb"],
        )
        groups[key].append(r)

    summary_rows = []
    for (
        cfg_id,
        label,
        method,
        objective_mode,
        objective_policy,
        delta,
        card,
        ic,
        sb,
    ), grp in sorted(groups.items()):
        n = len(grp)
        opt  = [r for r in grp if r["status"] == "OPTIMUM"]
        to   = [r for r in grp if r["status"] == "TIMEOUT"]
        uns  = [r for r in grp if r["status"] in ("UNSATISFIABLE", "UNSAT")]
        err  = [r for r in grp if r["status"] not in ("OPTIMUM","TIMEOUT","UNSATISFIABLE","UNSAT")]

        elapsed_opt  = [safe_float(r["elapsed_s"], 0) for r in opt]
        solve_opt    = [safe_float(r["solve_s"],   0) for r in opt]
        calls_all    = [safe_float(r["solver_calls"], 0) for r in grp]
        references   = [safe_float(r["similarity_reference_optimum"]) for r in opt if r.get("similarity_reference_optimum") is not None]
        lower_bounds = [safe_float(r["similarity_lower_bound"]) for r in opt if r.get("similarity_lower_bound") is not None]
        losses       = [safe_float(r["similarity_realized_loss_absolute"]) for r in opt if r.get("similarity_realized_loss_absolute") is not None]
        loss_ratios  = [safe_float(r["similarity_realized_loss_ratio"]) for r in opt if r.get("similarity_realized_loss_ratio") is not None]
        vars_all     = [safe_float(r["variables"],  0) for r in grp if r.get("variables")]
        hard_all     = [safe_float(r["hard_clauses"],0) for r in grp if r.get("hard_clauses")]
        soft_all     = [safe_float(r["soft_clauses"],0) for r in grp if r.get("soft_clauses")]
        best_opt     = [safe_float(r["best_value"])    for r in opt if r.get("best_value") is not None]
        sim_opt      = [safe_float(r["similarity"])    for r in opt if r.get("similarity")  is not None]
        cont_opt     = [safe_float(r["continuity"])    for r in opt if r.get("continuity")  is not None]
        ot_opt       = [safe_float(r["overtime"])      for r in opt if r.get("overtime")    is not None]

        def avg(lst): return round(mean(lst), 4) if lst else ""
        def med(lst): return round(median(lst), 4) if lst else ""
        def sd(lst):  return round(stdev(lst), 4) if len(lst) > 1 else ""

        par2 = round(compute_par2(grp), 4)

        summary_rows.append({
            "cfg_id":          cfg_id,
            "label":           label,
            "method":          method,
            "objective_mode":  objective_mode,
            "objective_policy": objective_policy,
            "delta":           delta,
            "cardinality":     card,
            "ic":              ic,
            "sb":              sb,
            "total_runs":      n,
            "optimum_n":       len(opt),
            "timeout_n":       len(to),
            "unsat_n":         len(uns),
            "error_n":         len(err),
            "optimum_pct":     round(100 * len(opt) / n, 1) if n else "",
            "mean_elapsed_s":  avg(elapsed_opt),
            "median_elapsed_s": med(elapsed_opt),
            "par2_s":          par2,
            "mean_solve_s":    avg(solve_opt),
            "mean_solver_calls": avg(calls_all),
            "mean_variables":  avg(vars_all),
            "mean_hard_clauses": avg(hard_all),
            "mean_soft_clauses": avg(soft_all),
            "mean_similarity_reference": avg(references),
            "mean_similarity_lower_bound": avg(lower_bounds),
            "mean_similarity_realized_loss": avg(losses),
            "mean_similarity_realized_loss_ratio": avg(loss_ratios),
            "mean_best_value": avg(best_opt),
            "mean_similarity": avg(sim_opt),
            "mean_continuity": avg(cont_opt),
            "mean_overtime":   avg(ot_opt),
            "std_best_value":  sd(best_opt),
        })

    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS,
                           extrasaction="ignore", delimiter=";")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Written: {out_path}  ({len(summary_rows)} rows)")


def write_instance_comparison_csv(rows: list[dict], out_path: Path) -> None:
    """Pivot: one row per instance and one block per policy/configuration."""
    def series_name(row: dict) -> str:
        delta_suffix = f'__delta_{row["delta"]}' if row["method"] == "epsilon" else ""
        return f'{row["method"]}{delta_suffix}__{row["label"]}'

    series_names = sorted(set(series_name(r) for r in rows))
    instances = sorted(set(r["instance_name"] for r in rows))

    # Build pivot dict
    pivot: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        iname = r["instance_name"]
        series = series_name(r)
        pivot[iname][series + "_status"] = r["status"]
        pivot[iname][series + "_weighted_reference"] = str(
            r.get("weighted_reference_score", "")
        )
        pivot[iname][series + "_stages"] = str(r.get("stage_optima", ""))
        pivot[iname][series + "_elapsed_s"] = str(
            round(safe_float(r.get("elapsed_s"), 0), 3)
        )
        pivot[iname][series + "_vars"] = str(r.get("variables", ""))

    # Column order
    cols = ["instance_name"]
    for series in series_names:
        cols += [
            f"{series}_status",
            f"{series}_weighted_reference",
            f"{series}_stages",
            f"{series}_elapsed_s",
            f"{series}_vars",
        ]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore",
                           delimiter=";", restval="")
        w.writeheader()
        for iname in instances:
            row = {"instance_name": iname}
            row.update(pivot[iname])
            w.writerow(row)
    print(
        f"  Written: {out_path}  "
        f"({len(instances)} instances × {len(series_names)} policy/config series)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} RESULT_DIR", file=sys.stderr)
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    if not result_dir.is_dir():
        print(f"ERROR: Not a directory: {result_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== Collecting results from: {result_dir} ===")

    rows = collect_results(result_dir)
    print(f"  Parsed {len(rows)} result files")

    if not rows:
        print("  No JSON results found. Run the experiment first.", file=sys.stderr)
        sys.exit(0)

    write_per_instance_csv(rows, result_dir / "results_per_instance.csv")
    write_summary_csv(rows, result_dir / "summary_by_config.csv")
    write_instance_comparison_csv(rows, result_dir / "comparison_pivot.csv")

    print()
    print("=== Output files ===")
    print(f"  {result_dir}/results_per_instance.csv   — 1 row per run")
    print(f"  {result_dir}/summary_by_config.csv      — 1 row per config (stats)")
    print(f"  {result_dir}/comparison_pivot.csv       — pivot: 1 row per instance")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
