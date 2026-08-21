"""Canonical constants for the ICIIT 2027 publication campaign.

The JSON manifest remains the machine-readable execution plan.  This module is
the independent, reviewable contract against which validators compare that
manifest, and it supplies shared configuration identities to analyzers and the
manuscript freeze step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments/configs/reduced_campaign_manifest.json"
CONTRACT_REVISION = "ICIIT 2027 compact campaign, revision 2026-08-22-v5"

REFERENCE_CONFIGURATION = ("totalizer", "both", "slot-service")
REFERENCE_CONFIGURATION_JSON = {
    "cardinality": REFERENCE_CONFIGURATION[0],
    "implied": REFERENCE_CONFIGURATION[1],
    "symmetry": REFERENCE_CONFIGURATION[2],
}
FACTORIAL_CONFIGURATIONS = tuple(
    (cardinality, implied, symmetry)
    for cardinality in ("sorting-network", "totalizer")
    for implied in ("none", "both")
    for symmetry in ("none", "slot-service")
)
FACTORIAL_CONFIGURATIONS_JSON = [
    {"cardinality": cardinality, "implied": implied, "symmetry": symmetry}
    for cardinality, implied, symmetry in FACTORIAL_CONFIGURATIONS
]

MAXSAT_SOLVER = {
    "name": "EvalMaxSAT",
    "platform": "linux-x86_64",
    "sha256": "97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7",
}

# (scientific name, config filename, expected runs, timeout seconds)
MEASURED_CAMPAIGNS = (
    ("original_factorial_ablation", "gcp_original_ablation.json", 384, 300),
    ("original_lex_cos_primary", "gcp_original_lex_primary.json", 84, 300),
    ("corrected_v2_policy_and_priority", "gcp_corrected_primary.json", 144, 300),
    ("commercial_exact_validation", "gcp_commercial_original.json", 80, 300),
    ("maxsat_commercial_validation", "gcp_maxsat_commercial_validation.json", 40, 300),
    (
        "corrected_v2_gurobi_policy_evidence",
        "gcp_commercial_corrected_primary.json",
        144,
        300,
    ),
    (
        "corrected_v2_cplex_stratum_audit",
        "gcp_commercial_corrected_audit.json",
        48,
        300,
    ),
)
NON_MEASURED_CAMPAIGNS = (
    ("evalmaxsat_lex_calibration", "gcp_evalmaxsat_lex_calibration.json", 4, 300),
    (
        "commercial_correctness_smoke",
        "gcp_commercial_correctness_smoke.json",
        18,
        30,
    ),
    (
        "corrected_v2_commercial_calibration",
        "gcp_commercial_corrected_calibration.json",
        48,
        300,
    ),
)
CORRECTED_EXACT_MEASURED_CAMPAIGNS = {
    "corrected_v2_gurobi_policy_evidence",
    "corrected_v2_cplex_stratum_audit",
}

EXPECTED_MEASURED_RUNS = sum(item[2] for item in MEASURED_CAMPAIGNS)
EXPECTED_WORST_CASE_SECONDS = sum(item[2] * item[3] for item in MEASURED_CAMPAIGNS)
EXPECTED_WORST_CASE_CORE_HOURS = EXPECTED_WORST_CASE_SECONDS / 3600

CORRECTED_COMMERCIAL_CALIBRATION_GATES: dict[str, Any] = {
    "schema_version": 1,
    "expected_instances": 8,
    "expected_runs": 48,
    "expected_methods": ["weighted", "lex-cos", "lex-overtime"],
    "expected_backends": ["gurobi-mip", "cplex-mip"],
    "minimum_all_policy_optimum_instances_per_backend": 6,
    "maximum_status_disagreements": 0,
    "maximum_objective_disagreements": 0,
    "maximum_technical_errors": 0,
    "maximum_unverified_optima": 0,
}
CORRECTED_EXACT_EVIDENCE_GATES: dict[str, Any] = {
    "schema_version": 1,
    "expected_primary_instances": 48,
    "expected_primary_runs": 144,
    "expected_audit_instances": 16,
    "expected_audit_runs": 48,
    "expected_strata": 16,
    "expected_methods": ["weighted", "lex-cos", "lex-overtime"],
    "minimum_all_policy_optimum_instances": 36,
    "minimum_strata_with_two_all_policy_optimum_seeds": 12,
    "required_audit_optimum_groups": 48,
    "maximum_status_disagreements": 0,
    "maximum_objective_disagreements": 0,
    "maximum_technical_errors": 0,
    "maximum_unverified_optima": 0,
}
