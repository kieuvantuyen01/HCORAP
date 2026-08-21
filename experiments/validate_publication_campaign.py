#!/usr/bin/env python3
"""Validate the frozen ICIIT 2027 publication-campaign contract.

The generic manifest validator checks arithmetic consistency.  This validator
also locks the scientific choices that must not drift between the paper, the
GCP wrapper, and the JSON configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .validate_campaign_manifest import validate as validate_manifest
    from .publication_contract import (
        CONTRACT_REVISION,
        CORRECTED_COMMERCIAL_CALIBRATION_GATES,
        CORRECTED_EXACT_EVIDENCE_GATES,
        DEFAULT_MANIFEST,
        EXPECTED_MEASURED_RUNS,
        EXPECTED_WORST_CASE_SECONDS,
        FACTORIAL_CONFIGURATIONS_JSON,
        MAXSAT_SOLVER,
        MEASURED_CAMPAIGNS,
        NON_MEASURED_CAMPAIGNS,
        REFERENCE_CONFIGURATION_JSON,
    )
except ImportError:  # Executed directly rather than imported as a package.
    from validate_campaign_manifest import validate as validate_manifest
    from publication_contract import (
        CONTRACT_REVISION,
        CORRECTED_COMMERCIAL_CALIBRATION_GATES,
        CORRECTED_EXACT_EVIDENCE_GATES,
        DEFAULT_MANIFEST,
        EXPECTED_MEASURED_RUNS,
        EXPECTED_WORST_CASE_SECONDS,
        FACTORIAL_CONFIGURATIONS_JSON,
        MAXSAT_SOLVER,
        MEASURED_CAMPAIGNS,
        NON_MEASURED_CAMPAIGNS,
        REFERENCE_CONFIGURATION_JSON,
    )


REFERENCE = REFERENCE_CONFIGURATION_JSON
FACTORIAL = FACTORIAL_CONFIGURATIONS_JSON
EXPECTED_MEASURED = list(MEASURED_CAMPAIGNS)
EXPECTED_NON_MEASURED = list(NON_MEASURED_CAMPAIGNS)
EXPECTED_CALIBRATION_GATES = CORRECTED_COMMERCIAL_CALIBRATION_GATES
EXPECTED_EXACT_EVIDENCE_GATES = CORRECTED_EXACT_EVIDENCE_GATES


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _methods(config: dict[str, Any]) -> list[str]:
    return [str(run.get("method")) for run in config.get("runs", [])]


def _check_equal(
    errors: list[str], label: str, observed: Any, expected: Any
) -> None:
    if observed != expected:
        errors.append(f"{label}: observed {observed!r}, expected {expected!r}")


def validate(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    config_root = manifest_path.parent
    arithmetic = validate_manifest(manifest_path)
    errors = list(arithmetic["errors"])
    manifest = _load(manifest_path)

    measured_rows = [
        (
            campaign.get("name"),
            campaign.get("config"),
            campaign.get("expected_runs"),
            campaign.get("timeout_seconds"),
        )
        for campaign in manifest.get("measured_campaigns", [])
    ]
    smoke_rows = [
        (
            campaign.get("name"),
            campaign.get("config"),
            campaign.get("expected_runs"),
            campaign.get("timeout_seconds"),
        )
        for campaign in manifest.get("non_measured_campaigns", [])
    ]
    _check_equal(errors, "measured campaign matrix", measured_rows, EXPECTED_MEASURED)
    _check_equal(
        errors, "non-measured campaign matrix", smoke_rows, EXPECTED_NON_MEASURED
    )
    _check_equal(
        errors,
        "measured run total",
        manifest.get("expected_measured_runs"),
        EXPECTED_MEASURED_RUNS,
    )
    _check_equal(
        errors,
        "MaxSAT solver identity",
        manifest.get("maxsat_solver"),
        MAXSAT_SOLVER,
    )
    _check_equal(
        errors,
        "worst-case measured seconds",
        manifest.get("expected_worst_case_seconds"),
        EXPECTED_WORST_CASE_SECONDS,
    )

    configs: dict[str, dict[str, Any]] = {}
    for _, filename, expected_runs, timeout in EXPECTED_MEASURED + EXPECTED_NON_MEASURED:
        path = config_root / filename
        try:
            config = _load(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path}: {exc}")
            continue
        configs[filename] = config
        _check_equal(
            errors,
            f"{filename} expected_runs",
            config.get("expected_runs"),
            expected_runs,
        )
        _check_equal(errors, f"{filename} timeout", config.get("timeout_seconds"), timeout)
        _check_equal(errors, f"{filename} workers", config.get("workers"), 1)

    for filename, config in configs.items():
        if filename == "gcp_commercial_correctness_smoke.json":
            _check_equal(errors, f"{filename} hard grace", config.get("hard_grace_seconds"), 15)
        else:
            _check_equal(errors, f"{filename} hard grace", config.get("hard_grace_seconds"), 60)

    factorial = configs.get("gcp_original_ablation.json", {})
    _check_equal(errors, "factorial instances", factorial.get("expected_instances"), 48)
    _check_equal(errors, "factorial configurations", factorial.get("configurations"), FACTORIAL)
    _check_equal(errors, "factorial methods", _methods(factorial), ["weighted"])

    maxsat_filenames = {
        "gcp_original_ablation.json",
        "gcp_original_lex_primary.json",
        "gcp_corrected_primary.json",
        "gcp_maxsat_commercial_validation.json",
        "gcp_evalmaxsat_lex_calibration.json",
    }
    for filename in maxsat_filenames:
        _check_equal(
            errors,
            f"{filename} solver",
            configs.get(filename, {}).get("solver"),
            "${EVALMAXSAT_BIN}",
        )

    maxsat_specs = {
        "gcp_original_lex_primary.json": (42, ["weighted", "lex-cos"]),
        "gcp_corrected_primary.json": (
            48,
            ["weighted", "lex-cos", "lex-overtime"],
        ),
        "gcp_maxsat_commercial_validation.json": (20, ["weighted", "lex-cos"]),
        "gcp_evalmaxsat_lex_calibration.json": (4, ["lex-cos"]),
    }
    for filename, (instances, methods) in maxsat_specs.items():
        config = configs.get(filename, {})
        _check_equal(errors, f"{filename} instances", config.get("expected_instances"), instances)
        _check_equal(
            errors,
            f"{filename} reference configuration",
            config.get("configurations"),
            [REFERENCE],
        )
        _check_equal(errors, f"{filename} methods", _methods(config), methods)

    seed_specs = {
        "gcp_original_ablation.json": [1, 2, 3],
        "gcp_original_lex_primary.json": [1, 2, 3],
        "gcp_corrected_primary.json": [1001, 1002, 1003],
        "gcp_maxsat_commercial_validation.json": list(range(1, 11)),
        "gcp_commercial_original.json": list(range(1, 11)),
        "gcp_commercial_corrected_primary.json": [1001, 1002, 1003],
        "gcp_commercial_corrected_audit.json": [1002],
        "gcp_commercial_corrected_calibration.json": [4],
    }
    for filename, seeds in seed_specs.items():
        observed = configs.get(filename, {}).get("instance_filters", {}).get("seeds")
        _check_equal(errors, f"{filename} evaluation seeds", observed, seeds)

    commercial_specs = {
        "gcp_commercial_original.json": [
            {"backend": "gurobi-mip", "formulation": "mip-e"},
            {"backend": "cplex-mip", "formulation": "mip-e"},
        ],
        "gcp_commercial_correctness_smoke.json": [
            {"backend": "gurobi-mip", "formulation": "mip-e"},
            {"backend": "cplex-mip", "formulation": "mip-e"},
            {
                "backend": "reference-enumerator",
                "formulation": "direct-schedule-enumeration",
            },
        ],
        "gcp_commercial_corrected_calibration.json": [
            {"backend": "gurobi-mip", "formulation": "mip-e"},
            {"backend": "cplex-mip", "formulation": "mip-e"},
        ],
        "gcp_commercial_corrected_primary.json": [
            {"backend": "gurobi-mip", "formulation": "mip-e"},
        ],
        "gcp_commercial_corrected_audit.json": [
            {"backend": "cplex-mip", "formulation": "mip-e"},
        ],
    }
    for filename, backends in commercial_specs.items():
        config = configs.get(filename, {})
        _check_equal(
            errors,
            f"{filename} backends",
            config.get("commercial_configurations"),
            backends,
        )
        expected_methods = (
            ["weighted", "lex-cos", "lex-overtime"]
            if filename.startswith("gcp_commercial_corrected_")
            else ["weighted", "lex-cos"]
        )
        _check_equal(errors, f"{filename} methods", _methods(config), expected_methods)
        for key, expected in (
            ("threads", 1),
            ("seed", 0),
            ("mip_gap", 0),
            ("absolute_mip_gap", 0),
        ):
            _check_equal(errors, f"{filename} {key}", config.get(key), expected)

    corrected_commercial_specs = {
        "gcp_commercial_corrected_calibration.json": (8, 48),
        "gcp_commercial_corrected_primary.json": (48, 144),
        "gcp_commercial_corrected_audit.json": (16, 48),
    }
    for filename, (instances, runs) in corrected_commercial_specs.items():
        config = configs.get(filename, {})
        _check_equal(errors, f"{filename} instances", config.get("expected_instances"), instances)
        _check_equal(errors, f"{filename} runs", config.get("expected_runs"), runs)
        _check_equal(
            errors,
            f"{filename} load profile",
            config.get("instance_filters", {}).get("load_profiles"),
            ["critical"],
        )
        _check_equal(
            errors,
            f"{filename} stores assignments and native logs",
            [
                (run.get("print_assignments"), run.get("native_log"))
                for run in config.get("runs", [])
            ],
            [(True, True), (True, True), (True, True)],
        )

    gate_specs = {
        "corrected_commercial_calibration_gates.json": EXPECTED_CALIBRATION_GATES,
        "corrected_exact_evidence_gates.json": EXPECTED_EXACT_EVIDENCE_GATES,
    }
    for filename, expected in gate_specs.items():
        try:
            observed = _load(config_root / filename)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {config_root / filename}: {exc}")
            continue
        _check_equal(errors, f"{filename} locked thresholds", observed, expected)

    return {
        **arithmetic,
        "valid": not errors,
        "contract": CONTRACT_REVISION,
        "measured_timeout_seconds": 300,
        "smoke_timeout_seconds": 30,
        "maxsat_solver": manifest.get("maxsat_solver"),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    try:
        report = validate(arguments.manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
