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
except ImportError:  # Executed directly rather than imported as a package.
    from validate_campaign_manifest import validate as validate_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments/configs/reduced_campaign_manifest.json"
REFERENCE = {
    "cardinality": "totalizer",
    "implied": "both",
    "symmetry": "slot-service",
}
FACTORIAL = [
    {"cardinality": cardinality, "implied": implied, "symmetry": symmetry}
    for cardinality in ("sorting-network", "totalizer")
    for implied in ("none", "both")
    for symmetry in ("none", "slot-service")
]
EXPECTED_MEASURED = [
    ("original_factorial_ablation", "gcp_original_ablation.json", 384, 300),
    ("original_lex_cos_primary", "gcp_original_lex_primary.json", 84, 300),
    ("corrected_v2_policy_and_priority", "gcp_corrected_primary.json", 144, 300),
    ("commercial_exact_validation", "gcp_commercial_original.json", 80, 300),
    (
        "maxsat_commercial_validation",
        "gcp_maxsat_commercial_validation.json",
        40,
        300,
    ),
]
EXPECTED_NON_MEASURED = [
    (
        "evalmaxsat_lex_calibration",
        "gcp_evalmaxsat_lex_calibration.json",
        4,
        300,
    ),
    (
        "commercial_correctness_smoke",
        "gcp_commercial_correctness_smoke.json",
        18,
        30,
    )
]


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
    _check_equal(errors, "measured run total", manifest.get("expected_measured_runs"), 732)
    _check_equal(
        errors,
        "MaxSAT solver identity",
        manifest.get("maxsat_solver"),
        {
            "name": "EvalMaxSAT",
            "platform": "linux-x86_64",
            "sha256": "97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7",
        },
    )
    _check_equal(
        errors,
        "worst-case measured seconds",
        manifest.get("expected_worst_case_seconds"),
        219600,
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
        _check_equal(errors, f"{filename} expected_runs", config.get("expected_runs"), expected_runs)
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
        _check_equal(errors, f"{filename} reference configuration", config.get("configurations"), [REFERENCE])
        _check_equal(errors, f"{filename} methods", _methods(config), methods)

    seed_specs = {
        "gcp_original_ablation.json": [1, 2, 3],
        "gcp_original_lex_primary.json": [1, 2, 3],
        "gcp_corrected_primary.json": [1001, 1002, 1003],
        "gcp_maxsat_commercial_validation.json": list(range(1, 11)),
        "gcp_commercial_original.json": list(range(1, 11)),
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
    }
    for filename, backends in commercial_specs.items():
        config = configs.get(filename, {})
        _check_equal(errors, f"{filename} backends", config.get("commercial_configurations"), backends)
        _check_equal(errors, f"{filename} methods", _methods(config), ["weighted", "lex-cos"])
        for key, expected in (
            ("threads", 1),
            ("seed", 0),
            ("mip_gap", 0),
            ("absolute_mip_gap", 0),
        ):
            _check_equal(errors, f"{filename} {key}", config.get(key), expected)

    return {
        **arithmetic,
        "valid": not errors,
        "contract": "ICIIT 2027 compact campaign, revision 2026-08-20-v3",
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
