#!/usr/bin/env python3
"""Create a non-destructive, reproducible audit of ``results_addition``.

The source result directories are treated as immutable imported data.  This
script writes only to ``results_addition/organized`` (or ``--output``) and
builds catalogs, checksums, consistency checks, and a publication-suitability
report that point back to the original files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ALLOWED_STATUSES = {
    "OPTIMUM",
    "UNSAT",
    "UNSATISFIABLE",
    "INFEASIBLE",
    "TIMEOUT",
    "TIMEOUT_FEASIBLE",
}


CAMPAIGN_REGISTRY: dict[str, dict[str, Any]] = {
    "commercial_30_15_4_40_25_5": {
        "role": "development-commercial-benchmark",
        "publication_decision": "exclude-primary",
        "expected_json": 400,
        "expected_basis": "4 configurations x 100 instances; corroborated by CSV and .done_runs",
        "reusable_scope": "development diagnostics only",
        "reasons": [
            "only 284 raw JSON files remain for 400 summarized rows",
            "benchmark classes 30_15_4 and 40_25_5 were used during development",
            "run predates the clean publication tag and the Ubuntu 24.04 GCP protocol",
        ],
    },
    "commercial_all_modes_30_15_4_40_25_5": {
        "role": "development-commercial-multiobjective",
        "publication_decision": "exclude-primary",
        "expected_json": 3200,
        "expected_basis": "4 configurations x 100 instances x 8 objective settings",
        "reusable_scope": "development and solver-consistency diagnostics only",
        "reasons": [
            "campaign is incomplete",
            "legacy lex-continuity/lex-overtime policies are not the frozen LEX-COS protocol",
            "benchmark classes were used during development and environment provenance is not publication-grade",
        ],
    },
    "commercial_main": {
        "role": "commercial-correctness-development",
        "publication_decision": "exclude-primary",
        "expected_json": 160,
        "expected_basis": "160 unique manifest run IDs after removing an appended duplicate pass",
        "reusable_scope": "software correctness smoke tests only",
        "reasons": [
            "manifest contains two copies of every run",
            "environment records five dirty Git files",
            "inputs are tests/instances rather than the paper benchmark",
        ],
    },
    "epsilon_8cfg_evalmaxsat": {
        "role": "legacy-epsilon-exploration",
        "publication_decision": "exclude-primary",
        "expected_json": 10000,
        "expected_basis": "5 deltas x 8 configurations x 250 benchmark instances (inferred from main campaign)",
        "reusable_scope": "historical exploratory diagnostics only",
        "reasons": [
            "run predates the clean publication tag and current EvalMaxSAT smoke/hash contract",
            "per-configuration campaign limits censor the benchmark in file-name order",
            "configuration naming is inconsistent in delta_0 and source commits differ across deltas",
        ],
    },
    "gcp_primary_analysis": {
        "role": "empty-placeholder",
        "publication_decision": "no-evidence",
        "expected_json": None,
        "expected_basis": "no campaign contract found",
        "reusable_scope": "none",
        "reasons": ["directory is empty"],
    },
    "iciit2027_all_solvers": {
        "role": "aborted-all-solvers-development-run",
        "publication_decision": "exclude-primary",
        "expected_json": 2400,
        "expected_basis": "explicit run.log contract: 3 solvers x 8 settings x 100 instances",
        "reusable_scope": "first-20-instance EvalMaxSAT smoke evidence only",
        "reasons": [
            "only one of the three named solvers started and only 20 JSON files were completed",
            "run predates the clean publication tag and current campaign matrix",
            "cardinality/implied/symmetry settings do not represent the proposed configuration",
        ],
    },
    "lex_8cfg_evalmaxsat": {
        "role": "legacy-lexicographic-exploration",
        "publication_decision": "exclude-primary",
        "expected_json": 4000,
        "expected_basis": "2 policies x 8 configurations x 250 benchmark instances (inferred from main campaign)",
        "reusable_scope": "historical exploratory diagnostics only",
        "reasons": [
            "run predates the clean publication tag and current campaign matrix",
            "policies predate the frozen LEX-COS/LEX-OCS naming and ordering",
            "per-configuration campaign limits produce partial, sometimes unbalanced coverage",
        ],
    },
    "main_8cfg_evalmaxsat": {
        "role": "legacy-eight-configuration-ablation",
        "publication_decision": "exclude-primary",
        "expected_json": 2000,
        "expected_basis": "8 configurations x 250 benchmark instances",
        "reusable_scope": "historical encoding diagnostics only",
        "reasons": [
            "run predates the clean publication tag and current campaign matrix",
            "four raw results are absent from the balanced 8 x 250 matrix",
            "run predates the clean publication tag and pinned EvalMaxSAT protocol",
        ],
    },
    "paper_test": {
        "role": "single-instance-commercial-smoke",
        "publication_decision": "exclude-primary",
        "expected_json": 17,
        "expected_basis": "17 unique manifest rows for one benchmark instance",
        "reusable_scope": "software smoke test only",
        "reasons": [
            "contains one benchmark instance",
            "environment records six dirty Git files",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def campaign_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if relative.parts else ""


def read_environment(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("{"):
            continue
        key, value = line.split("=", 1)
        if key and key.replace("_", "").isalnum():
            values[key] = value
    return values


def nearest_environment(root: Path, path: Path) -> tuple[str | None, dict[str, Any]]:
    current = path.parent
    while current == root or root in current.parents:
        candidate = current / "environment.txt"
        if candidate.is_file():
            return candidate.relative_to(root).as_posix(), read_environment(candidate)
        if current == root:
            break
        current = current.parent
    return None, {}


def analysis_unit_and_config(
    root: Path, path: Path, payload: dict[str, Any]
) -> tuple[str, str]:
    parts = path.relative_to(root).parts
    campaign = parts[0]
    if campaign in {
        "epsilon_8cfg_evalmaxsat",
        "lex_8cfg_evalmaxsat",
        "commercial_all_modes_30_15_4_40_25_5",
        "iciit2027_all_solvers",
    } and len(parts) >= 4:
        return f"{campaign}/{parts[1]}", parts[2]
    if campaign in {
        "main_8cfg_evalmaxsat",
        "commercial_30_15_4_40_25_5",
    } and len(parts) >= 3:
        return campaign, parts[1]
    backend = payload.get("backend") or payload.get("solver") or "unknown"
    formulation = payload.get("formulation")
    config = f"{backend}/{formulation}" if formulation else str(backend)
    return campaign, config


def objective_signature(payload: dict[str, Any]) -> str | None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    mode = payload.get("objective_mode") or payload.get("method")
    if mode == "weighted":
        signature = [metrics.get("coverage"), metrics.get("weighted_reference_score")]
        return json.dumps(signature, separators=(",", ":"))
    stages = payload.get("stages")
    stage_values = []
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            stage_values.append(
                [
                    stage.get("objective") or stage.get("name"),
                    stage.get("optimum", stage.get("incumbent")),
                ]
            )
    if stage_values:
        return json.dumps(
            [metrics.get("coverage"), stage_values], separators=(",", ":")
        )
    return json.dumps(
        [
            metrics.get("coverage"),
            metrics.get("similarity"),
            metrics.get("continuity"),
            metrics.get("overtime"),
        ],
        separators=(",", ":"),
    )


def normalized_method(payload: dict[str, Any], path: Path) -> str:
    method = payload.get("method")
    if method:
        return str(method)
    mode = payload.get("objective_mode")
    if mode:
        return str(mode)
    for part in path.parts:
        if part.startswith("lex-") or part.startswith("epsilon_delta_"):
            return part
    return "unknown"


def json_record(root: Path, path: Path, file_hash: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    unit, config = analysis_unit_and_config(root, path, payload)
    environment_path, environment = nearest_environment(root, path)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    instance = payload.get("instance")
    status = payload.get("status")
    delta = payload.get("delta")
    method = normalized_method(payload, path)
    elapsed = payload.get("elapsed_seconds")
    timeout = payload.get("timeout_seconds")
    formula_ok: bool | None = None
    if metrics and all(
        isinstance(metrics.get(key), (int, float))
        for key in ("similarity", "continuity", "overtime", "weighted_reference_score")
    ):
        wc = payload.get("continuity_weight", 1)
        wo = payload.get("overtime_weight", 1)
        if isinstance(wc, (int, float)) and isinstance(wo, (int, float)):
            expected_score = (
                metrics["similarity"]
                - wc * metrics["continuity"]
                - wo * metrics["overtime"]
            )
            formula_ok = abs(expected_score - metrics["weighted_reference_score"]) < 1e-9
    timeout_overrun = None
    if isinstance(elapsed, (int, float)) and isinstance(timeout, (int, float)) and timeout > 0:
        timeout_overrun = elapsed > timeout * 1.05
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_hash,
        "campaign": campaign_name(root, path),
        "analysis_unit": unit,
        "configuration": config,
        "schema_version": payload.get("schema_version"),
        "instance": instance,
        "instance_name": Path(str(instance)).stem if instance else path.stem,
        "status": status,
        "method": method,
        "objective_mode": payload.get("objective_mode"),
        "objective_policy": payload.get("objective_policy"),
        "delta": None if delta is None else str(delta),
        "backend": payload.get("backend"),
        "formulation": payload.get("formulation"),
        "solver": payload.get("solver"),
        "solver_version": payload.get("solver_version"),
        "cardinality_encoding": payload.get("cardinality_encoding"),
        "implied_constraints": payload.get("implied_constraints"),
        "symmetry_breaking": payload.get("symmetry_breaking"),
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout,
        "timeout_overrun_gt_5pct": timeout_overrun,
        "solver_calls": payload.get("solver_calls"),
        "verified": metrics.get("verified") if metrics else None,
        "coverage": metrics.get("coverage") if metrics else None,
        "similarity": metrics.get("similarity") if metrics else None,
        "continuity": metrics.get("continuity") if metrics else None,
        "overtime": metrics.get("overtime") if metrics else None,
        "weighted_reference_score": (
            metrics.get("weighted_reference_score") if metrics else None
        ),
        "weighted_score_formula_ok": formula_ok,
        "objective_signature": objective_signature(payload),
        "error": payload.get("error"),
        "environment_path": environment_path,
        "source_git_commit": environment.get("git_commit"),
        "binary_sha256": environment.get("hcorap_sha256")
        or environment.get("binary_sha256"),
        "solver_sha256": environment.get("solver_sha256"),
        "environment_created_utc": environment.get("created_utc"),
    }


def read_delimited(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if path.suffix == ".tsv" else (";" if first_line.count(";") > first_line.count(",") else ",")
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    return list(reader.fieldnames or []), list(reader)


def audit_result_csvs(root: Path, json_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records_by_prefix: Counter[str] = Counter()
    for record in json_records:
        parts = Path(record["path"]).parts
        for length in range(1, len(parts)):
            records_by_prefix[Path(*parts[:length]).as_posix()] += 1
    audits = []
    for path in sorted(root.rglob("results_per_instance.csv")):
        fields, rows = read_delimited(path)
        relative_parent = path.parent.relative_to(root).as_posix()
        statuses = Counter(row.get("status") or "[missing]" for row in rows)
        audits.append(
            {
                "path": path.relative_to(root).as_posix(),
                "rows": len(rows),
                "columns": fields,
                "status_counts": dict(sorted(statuses.items())),
                "json_files_in_scope": records_by_prefix[relative_parent],
                "row_minus_json": len(rows) - records_by_prefix[relative_parent],
            }
        )
    return audits


def audit_done_files(root: Path) -> list[dict[str, Any]]:
    audits = []
    for path in sorted(root.rglob(".done_runs")):
        rows = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line]
        audits.append(
            {
                "path": path.relative_to(root).as_posix(),
                "rows": len(rows),
                "unique_rows": len(set(rows)),
                "duplicate_rows": len(rows) - len(set(rows)),
            }
        )
    return audits


def audit_manifests(root: Path) -> list[dict[str, Any]]:
    audits = []
    for path in sorted(root.rglob("manifest.tsv")):
        fields, rows = read_delimited(path)
        run_ids = [row.get("run_id", "") for row in rows]
        result_paths = [row.get("result", "") for row in rows]
        existing = 0
        for result in set(result_paths):
            if not result:
                continue
            if (path.parent / Path(result).name).is_file():
                existing += 1
        audits.append(
            {
                "path": path.relative_to(root).as_posix(),
                "columns": fields,
                "rows": len(rows),
                "unique_run_ids": len(set(run_ids)),
                "duplicate_run_id_rows": len(run_ids) - len(set(run_ids)),
                "unique_result_paths": len(set(result_paths)),
                "duplicate_result_path_rows": len(result_paths) - len(set(result_paths)),
                "unique_result_basenames_present": existing,
            }
        )
    return audits


def group_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["analysis_unit"], record["configuration"])].append(record)
    summaries = []
    for (unit, config), rows in sorted(grouped.items()):
        statuses = Counter(row["status"] or "[missing]" for row in rows)
        instances = {row["instance_name"] for row in rows}
        summaries.append(
            {
                "analysis_unit": unit,
                "configuration": config,
                "runs": len(rows),
                "unique_instances": len(instances),
                "status_counts": dict(sorted(statuses.items())),
                "verified_true": sum(row["verified"] is True for row in rows),
                "verified_false": sum(row["verified"] is False for row in rows),
                "verified_missing": sum(row["verified"] is None for row in rows),
                "errors_nonempty": sum(bool(row["error"]) for row in rows),
            }
        )
    return summaries


def coverage_balance(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in records:
        cell_key = (row["analysis_unit"], row["method"], row["delta"] or "-")
        cells[cell_key][row["configuration"]].add(row["instance_name"])
    audits = []
    for (unit, method, delta), configs in sorted(cells.items()):
        instance_sets = list(configs.values())
        union = set().union(*instance_sets) if instance_sets else set()
        intersection = set.intersection(*instance_sets) if instance_sets else set()
        counts = {key: len(value) for key, value in sorted(configs.items())}
        audits.append(
            {
                "analysis_unit": unit,
                "method": method,
                "delta": delta,
                "configurations": len(configs),
                "instance_counts_by_configuration": counts,
                "union_instances": len(union),
                "intersection_instances": len(intersection),
                "balanced_instance_sets": all(value == instance_sets[0] for value in instance_sets[1:]) if instance_sets else True,
            }
        )
    return audits


def agreement_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["analysis_unit"], row["method"], row["delta"] or "-", row["instance_name"])].append(row)
    summary: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "multi_configuration_instances": 0,
            "all_optimum_instances": 0,
            "objective_disagreements": 0,
            "proved_status_disagreements": 0,
            "examples": [],
        }
    )
    proved = {"OPTIMUM", "UNSAT", "UNSATISFIABLE", "INFEASIBLE"}
    for (unit, method, delta, instance), rows in sorted(grouped.items()):
        configs = {row["configuration"] for row in rows}
        if len(configs) < 2:
            continue
        item = summary[(unit, method, delta)]
        item["multi_configuration_instances"] += 1
        statuses = {row["status"] for row in rows}
        proved_statuses = statuses & proved
        if len(proved_statuses) > 1:
            item["proved_status_disagreements"] += 1
            if len(item["examples"]) < 5:
                item["examples"].append({"instance": instance, "statuses": sorted(statuses)})
        if all(row["status"] == "OPTIMUM" for row in rows):
            item["all_optimum_instances"] += 1
            signatures = {row["objective_signature"] for row in rows}
            if len(signatures) > 1:
                item["objective_disagreements"] += 1
                if len(item["examples"]) < 5:
                    item["examples"].append(
                        {"instance": instance, "objective_signatures": sorted(str(value) for value in signatures)}
                    )
    return [
        {
            "analysis_unit": key[0],
            "method": key[1],
            "delta": key[2],
            **value,
        }
        for key, value in sorted(summary.items())
    ]


def campaign_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_campaign[row["campaign"]].append(row)
    names = sorted(set(CAMPAIGN_REGISTRY) | set(by_campaign))
    output = []
    for name in names:
        rows = by_campaign.get(name, [])
        policy = CAMPAIGN_REGISTRY.get(
            name,
            {
                "role": "unclassified",
                "publication_decision": "manual-review",
                "expected_json": None,
                "expected_basis": "none",
                "reusable_scope": "manual review required",
                "reasons": ["campaign is not registered"],
            },
        )
        expected = policy["expected_json"]
        actual = len(rows)
        output.append(
            {
                "campaign": name,
                **policy,
                "actual_json": actual,
                "completion_fraction": actual / expected if expected else None,
                "missing_vs_expected": expected - actual if expected is not None else None,
                "status_counts": dict(sorted(Counter(row["status"] or "[missing]" for row in rows).items())),
                "source_git_commits": sorted({row["source_git_commit"] for row in rows if row["source_git_commit"]}),
                "binary_sha256_values": sorted({row["binary_sha256"] for row in rows if row["binary_sha256"]}),
                "solver_sha256_values": sorted({row["solver_sha256"] for row in rows if row["solver_sha256"]}),
            }
        )
    return output


def markdown_report(
    generated_at: str,
    inventory: list[dict[str, Any]],
    records: list[dict[str, Any]],
    campaigns: list[dict[str, Any]],
    csv_audits: list[dict[str, Any]],
    done_audits: list[dict[str, Any]],
    manifest_audits: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    agreements: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
) -> str:
    invalid_status = sum(row["status"] not in ALLOWED_STATUSES for row in records)
    formula_mismatch = sum(row["weighted_score_formula_ok"] is False for row in records)
    timeout_overruns = sum(row["timeout_overrun_gt_5pct"] is True for row in records)
    optimum = sum(row["status"] == "OPTIMUM" for row in records)
    optimum_unverified = sum(
        row["status"] == "OPTIMUM" and row["verified"] is not True
        for row in records
    )
    evalmaxsat = sum(
        "EvalMaxSAT" in str(row["solver"]) or "EVALMAXSAT" in row["configuration"]
        for row in records
    )
    unbalanced = [row for row in coverage if not row["balanced_instance_sets"]]
    agreement_mismatch = sum(row["objective_disagreements"] for row in agreements)
    proved_disagreement = sum(row["proved_status_disagreements"] for row in agreements)
    lines = [
        "# Audit kết quả bổ sung",
        "",
        f"Thời điểm tạo (UTC): `{generated_at}`.",
        "",
        "> Kết luận sử dụng: **không có campaign nào trong `results_addition/` đủ điều kiện đưa trực tiếp vào bảng kết quả chính của bản thảo theo submission protocol hiện tại.** Dữ liệu gốc được giữ nguyên để làm bằng chứng phát triển, smoke test và chẩn đoán.",
        "",
        "## Tóm tắt kiểm kê",
        "",
        f"- {len(inventory):,} file nguồn, tổng dung lượng {sum(row['size_bytes'] for row in inventory):,} byte.",
        f"- {len(records):,} JSON hợp lệ về cú pháp; {invalid_status} trạng thái ngoài vocabulary đã biết.",
        f"- {len(csv_audits)} bảng `results_per_instance.csv`, {len(done_audits)} sổ `.done_runs`, {len(manifest_audits)} manifest TSV.",
        f"- {len(duplicate_groups)} nhóm file trùng nội dung SHA-256; đây là trùng vật lý, tách biệt với dòng manifest/done-runs bị append lặp.",
        f"- {optimum:,} JSON mang trạng thái `OPTIMUM`; {optimum_unverified} trong số đó thiếu cờ kiểm chứng nghiệm.",
        f"- {formula_mismatch} nghiệm có weighted-reference formula không khớp; {timeout_overruns} lượt vượt timeout trên 5%.",
        f"- {len(unbalanced)} ô thí nghiệm có tập instance không cân bằng; {agreement_mismatch} nhóm all-optimum bất đồng objective signature; {proved_disagreement} nhóm bất đồng trạng thái đã chứng minh.",
        "",
        "## Phân loại campaign",
        "",
        "| Campaign | JSON / dự kiến | Trạng thái | Phạm vi có thể tái sử dụng |",
        "|---|---:|---|---|",
    ]
    for row in campaigns:
        expected = row["expected_json"]
        count = f"{row['actual_json']:,}" if expected is None else f"{row['actual_json']:,} / {expected:,}"
        lines.append(
            f"| `{row['campaign']}` | {count} | `{row['publication_decision']}` | {row['reusable_scope']} |"
        )
    lines.extend(["", "## Phát hiện bắt buộc xử lý", ""])
    findings = [
        f"**EvalMaxSAT lịch sử:** tổng cộng {evalmaxsat:,} JSON chỉ được giữ làm diagnostic vì khác commit, sampling và provenance. Publication campaign mới vẫn dùng EvalMaxSAT, nhưng phải chạy lại với binary hash và smoke contract đã khóa; không nhập trực tiếp runtime, optimum rate hay Pareto points cũ vào claim chính.",
        "**Commercial parser/raw mismatch:** `commercial_30_15_4_40_25_5` có 400 dòng CSV, gồm đúng 116 `PARSE_ERROR`, và 400 done markers nhưng chỉ 284 JSON. Không thể truy vết 116 dòng lỗi về raw result; đây chính là commercial dataset lịch sử bị parser loại sai.",
        "**Commercial campaign bị ngắt:** `commercial_all_modes_30_15_4_40_25_5` dự kiến 3.200 JSON nhưng có 3.149; `.done_runs` bị append lặp và CSV không đồng nhất với số JSON.",
        "**Manifest bị nhân đôi:** `commercial_main/manifest.tsv` có 320 dòng nhưng chỉ 160 run ID/result path duy nhất. Các JSON tồn tại, song đây là correctness test trên `tests/instances` và source tree được ghi nhận là dirty.",
        "**Censoring theo thứ tự file:** các campaign lex/epsilon dừng từng cấu hình theo time budget, nên tập instance có thể khác giữa cấu hình và bị phụ thuộc thứ tự tên file. Chỉ phân tích paired intersection nếu dùng cho chẩn đoán; không diễn giải như benchmark confirmatory.",
        "**Provenance không đồng nhất:** epsilon `delta_0` dùng commit/binary khác bốn delta còn lại; toàn bộ dữ liệu chạy trên Ubuntu 22.04 và không chứng minh đúng máy `c4-highcpu-8`, trong khi protocol hiện tại khóa Ubuntu 24.04, EvalMaxSAT SHA-256 và publication tag sạch.",
        "**Tên thư mục gây hiểu nhầm:** `iciit2027_all_solvers` mới có 20 EvalMaxSAT run của mode weighted; Gurobi và CPLEX chưa bắt đầu. Không được mô tả là all-solvers result.",
        "**Epsilon delta_0 có schema thư mục lệch:** xuất hiện cả `cfg1_ORIGINAL` và `cfg5_ORIGINAL`, khác thứ tự cấu hình của các delta còn lại. Việc gộp theo `cfg_id` sẽ sai nếu không chuẩn hóa bằng ba thuộc tính encoding/implied/symmetry.",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Quy tắc sử dụng cho bản thảo",
            "",
            "1. Không merge các CSV/XLSX hiện có vào `experiments/results/` của publication campaign.",
            "2. Chỉ trích xuất từ raw JSON khi làm chẩn đoán; luôn lọc theo semantic configuration, không theo `cfg_id`.",
            "3. Không dùng các aggregate epsilon (`epsilon_pareto_frontier.csv`, `epsilon_unique_points.csv`, ...) cho claim Pareto vì nguồn EvalMaxSAT bị loại và campaign bị censor.",
            "4. Kết quả Gurobi/CPLEX cũ có thể dùng để kiểm tra code nội bộ, nhưng bảng chính vẫn phải đến từ publication runner mới với raw JSON, manifest, validation và checksum đầy đủ.",
            "5. Giữ nguyên toàn bộ file nguồn. Mọi bảng dẫn xuất mới phải ghi input SHA-256 từ `checksums/SHA256SUMS` và script tạo bảng.",
            "",
            "## Cấu trúc đã tổ chức",
            "",
            "- `catalog/file_inventory.jsonl`: mọi file nguồn, kích thước và SHA-256.",
            "- `catalog/json_runs.jsonl`: chỉ mục chuẩn hóa cho từng JSON result.",
            "- `catalog/campaign_summary.json`: phân loại, completeness và provenance theo campaign.",
            "- `quality/*.json`: đối chiếu CSV/JSON, duplicate markers, balance và objective agreement.",
            "- `checksums/SHA256SUMS`: checksum chỉ cho dữ liệu nguồn, không tự bao gồm `organized/`.",
            "",
            "## Ghi chú spreadsheet",
            "",
            "Ba workbook XLSX trong `main_8cfg_evalmaxsat` được giữ nguyên như output lịch sử. Audit không tạo hoặc sửa workbook; catalog chuẩn được xuất JSON/JSONL để tránh hợp thức hóa các bảng dẫn xuất từ campaign không đủ điều kiện publication.",
            "",
        ]
    )
    return "\n".join(lines)


def organized_readme(generated_at: str) -> str:
    return f"""# Organized audit layer

Generated at `{generated_at}` by `experiments/audit_results_addition.py`.

This directory contains derived indexes and quality reports only.  The sibling
campaign directories are immutable source evidence; no raw result was renamed,
moved, deduplicated, or overwritten.

Rebuild from the repository root:

```bash
python3 experiments/audit_results_addition.py --root results_addition
```

The publication decision is documented in
`reports/RESULTS_ADDITION_AUDIT_20260819.md`.  Start there before using any
number from the imported result set.
"""


def audit(root: Path, output: Path, generated_at: str) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"results root does not exist: {root}")
    if output != root and root not in output.parents:
        raise ValueError("output must be inside the results root")

    source_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not (path == output or output in path.parents)
    )
    inventory = []
    file_hashes: dict[Path, str] = {}
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for path in source_files:
        digest = sha256(path)
        file_hashes[path] = digest
        relative = path.relative_to(root).as_posix()
        inventory.append(
            {
                "path": relative,
                "campaign": campaign_name(root, path),
                "suffix": path.suffix.lower() or "[none]",
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
        hash_groups[digest].append(relative)

    parse_errors = []
    records = []
    for path in (path for path in source_files if path.suffix.lower() == ".json"):
        try:
            records.append(json_record(root, path, file_hashes[path]))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            parse_errors.append(
                {"path": path.relative_to(root).as_posix(), "error": str(error)}
            )

    duplicate_groups = [
        {"sha256": digest, "size": len(paths), "paths": sorted(paths)}
        for digest, paths in sorted(hash_groups.items())
        if len(paths) > 1
    ]
    csv_audits = audit_result_csvs(root, records)
    done_audits = audit_done_files(root)
    manifest_audits = audit_manifests(root)
    groups = group_summaries(records)
    coverage = coverage_balance(records)
    agreements = agreement_audit(records)
    campaigns = campaign_summaries(records)

    summary = {
        "generated_at_utc": generated_at,
        "source_root": root.name,
        "source_files": len(inventory),
        "source_bytes": sum(row["size_bytes"] for row in inventory),
        "json_files": sum(path.suffix.lower() == ".json" for path in source_files),
        "json_records_parsed": len(records),
        "json_parse_errors": len(parse_errors),
        "duplicate_content_groups": len(duplicate_groups),
        "publication_eligible_campaigns": [
            row["campaign"]
            for row in campaigns
            if row["publication_decision"] == "publication-eligible"
        ],
    }

    write_jsonl(output / "catalog" / "file_inventory.jsonl", inventory)
    write_jsonl(output / "catalog" / "json_runs.jsonl", records)
    write_json(output / "catalog" / "campaign_summary.json", campaigns)
    write_json(output / "catalog" / "audit_summary.json", summary)
    write_json(output / "quality" / "json_parse_errors.json", parse_errors)
    write_json(output / "quality" / "duplicate_content.json", duplicate_groups)
    write_json(output / "quality" / "csv_json_consistency.json", csv_audits)
    write_json(output / "quality" / "done_runs_audit.json", done_audits)
    write_json(output / "quality" / "manifest_audit.json", manifest_audits)
    write_json(output / "quality" / "configuration_status.json", groups)
    write_json(output / "quality" / "instance_coverage_balance.json", coverage)
    write_json(output / "quality" / "objective_agreement.json", agreements)

    checksum_path = output / "checksums" / "SHA256SUMS"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in inventory),
        encoding="utf-8",
    )
    report = markdown_report(
        generated_at,
        inventory,
        records,
        campaigns,
        csv_audits,
        done_audits,
        manifest_audits,
        coverage,
        agreements,
        duplicate_groups,
    )
    report_path = output / "reports" / "RESULTS_ADDITION_AUDIT_20260819.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    (output / "README.md").write_text(organized_readme(generated_at), encoding="utf-8")
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results_addition"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--generated-at",
        help="fixed ISO timestamp for reproducible report builds",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    output = arguments.output.resolve() if arguments.output else root / "organized"
    generated_at = arguments.generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    summary = audit(root, output, generated_at)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
