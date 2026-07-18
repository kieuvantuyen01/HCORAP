#!/usr/bin/env python3
"""
collect_main_results.py
Tổng hợp kết quả thực nghiệm 8 cấu hình thành:
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
    "cardinality", "ic", "sb",
    "status",
    "elapsed_s", "solve_s",
    "variables", "hard_clauses", "soft_clauses",
    "best_value", "similarity", "continuity", "overtime", "coverage",
]

SUMMARY_COLS = [
    "cfg_id", "label", "cardinality", "ic", "sb",
    "total_runs",
    "optimum_n", "timeout_n", "unsat_n", "error_n",
    "optimum_pct",
    "mean_elapsed_s", "median_elapsed_s", "par2_s",
    "mean_solve_s",
    "mean_variables", "mean_hard_clauses", "mean_soft_clauses",
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
        return {"status": "PARSE_ERROR", "error": str(e)}

    status = d.get("status", "ERROR")
    elapsed = safe_float(d.get("elapsed_seconds"), 0.0)

    stages = d.get("stages") or []
    solve_s = sum(s.get("solve_seconds", 0) for s in stages)
    variables  = stages[0].get("variables", 0)   if stages else 0
    hard_cl    = stages[0].get("hard_clauses", 0) if stages else 0
    soft_cl    = stages[0].get("soft_clauses", 0) if stages else 0

    m = d.get("metrics") or {}
    sim  = safe_float(m.get("similarity"))
    cont = safe_float(m.get("continuity"))
    ot   = safe_float(m.get("overtime"))
    cov  = 120 if d.get("full_coverage") else safe_float(m.get("coverage"))

    wc = safe_float(d.get("continuity_weight"), 1.0)
    wo = safe_float(d.get("overtime_weight"),   1.0)
    P  = safe_float(d.get("P_value"), -1.0)

    best = None
    if sim is not None and cont is not None and ot is not None:
        best = sim - wc * cont - wo * abs(P) * ot

    return {
        "status":       status,
        "elapsed_s":    elapsed,
        "solve_s":      solve_s,
        "variables":    variables,
        "hard_clauses": hard_cl,
        "soft_clauses": soft_cl,
        "best_value":   best,
        "similarity":   sim,
        "continuity":   cont,
        "overtime":     ot,
        "coverage":     cov,
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
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_RUN_COLS,
                           extrasaction="ignore", delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {out_path}  ({len(rows)} rows)")


def compute_par2(times: list[float], statuses: list[str],
                 timeout: float = TIMEOUT_SENTINEL) -> float:
    """PAR-2: mỗi timeout tính 2×timeout."""
    vals = []
    for t, s in zip(times, statuses):
        if s == "OPTIMUM":
            vals.append(t)
        else:
            vals.append(2 * timeout)
    return mean(vals) if vals else float("nan")


def write_summary_csv(rows: list[dict], out_path: Path) -> None:
    # Group by (cfg_id, label, cardinality, ic, sb)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["cfg_id"], r["label"], r["cardinality"], r["ic"], r["sb"])
        groups[key].append(r)

    summary_rows = []
    for (cfg_id, label, card, ic, sb), grp in sorted(groups.items()):
        n = len(grp)
        opt  = [r for r in grp if r["status"] == "OPTIMUM"]
        to   = [r for r in grp if r["status"] == "TIMEOUT"]
        uns  = [r for r in grp if r["status"] in ("UNSATISFIABLE", "UNSAT")]
        err  = [r for r in grp if r["status"] not in ("OPTIMUM","TIMEOUT","UNSATISFIABLE","UNSAT")]

        elapsed_all  = [safe_float(r["elapsed_s"], 0) for r in grp]
        elapsed_opt  = [safe_float(r["elapsed_s"], 0) for r in opt]
        solve_opt    = [safe_float(r["solve_s"],   0) for r in opt]
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

        par2 = round(compute_par2(elapsed_all,
                                   [r["status"] for r in grp]), 4)

        summary_rows.append({
            "cfg_id":          cfg_id,
            "label":           label,
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
            "mean_variables":  avg(vars_all),
            "mean_hard_clauses": avg(hard_all),
            "mean_soft_clauses": avg(soft_all),
            "mean_best_value": avg(best_opt),
            "mean_similarity": avg(sim_opt),
            "mean_continuity": avg(cont_opt),
            "mean_overtime":   avg(ot_opt),
            "std_best_value":  sd(best_opt),
        })

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLS,
                           extrasaction="ignore", delimiter=";")
        w.writeheader()
        w.writerows(summary_rows)
    print(f"  Written: {out_path}  ({len(summary_rows)} rows)")


def write_instance_comparison_csv(rows: list[dict], out_path: Path) -> None:
    """Pivot: 1 row per instance, 8 cột best_value (một per cfg)."""
    # Collect all labels
    labels = sorted(set(r["label"] for r in rows))
    instances = sorted(set(r["instance_name"] for r in rows))

    # Build pivot dict
    pivot: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        iname = r["instance_name"]
        lbl   = r["label"]
        pivot[iname][lbl + "_status"]    = r["status"]
        pivot[iname][lbl + "_best"]      = str(r.get("best_value", ""))
        pivot[iname][lbl + "_elapsed_s"] = str(round(safe_float(r.get("elapsed_s"), 0), 3))
        pivot[iname][lbl + "_vars"]      = str(r.get("variables", ""))

    # Column order
    cols = ["instance_name"]
    for lbl in labels:
        cols += [f"{lbl}_status", f"{lbl}_best", f"{lbl}_elapsed_s", f"{lbl}_vars"]

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore",
                           delimiter=";", restval="")
        w.writeheader()
        for iname in instances:
            row = {"instance_name": iname}
            row.update(pivot[iname])
            w.writerow(row)
    print(f"  Written: {out_path}  ({len(instances)} instances × {len(labels)} configs)")


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
