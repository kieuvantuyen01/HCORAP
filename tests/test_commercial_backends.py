from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bin" / "release" / "hcorap_commercial"
TRADEOFF = ROOT / "tests" / "instances" / "tradeoff.txt"
PARTIAL = ROOT / "tests" / "instances" / "partial_coverage.txt"
SPARSE_USERS = ROOT / "tests" / "instances" / "sparse_users.txt"
LEX_COS_TIE = ROOT / "tests" / "instances" / "lex_cos_tie.txt"


def _require_binary() -> None:
    if not BINARY.is_file():
        pytest.skip("build with: make -j4 YICES=0 hcorap_commercial")


def _run(instance: Path, method: str, *extra: str) -> dict:
    _require_binary()
    completed = subprocess.run(
        [
            str(BINARY),
            str(instance),
            "--backend",
            "reference-enumerator",
            "--method",
            method,
            "--timeout",
            "10",
            *extra,
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=15,
    )
    return json.loads(completed.stdout)


def test_backend_inventory_is_machine_readable() -> None:
    _require_binary()
    completed = subprocess.run(
        [str(BINARY), "--list-backends"],
        check=True,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)
    backends = {row["name"]: row for row in result["backends"]}
    assert backends["gurobi-mip"]["formulations"] == ["mip-e"]
    assert backends["cplex-mip"]["formulations"] == ["mip-e"]
    assert backends["cplex-cp"]["formulations"] == ["cp-t", "cp-i"]
    assert backends["reference-enumerator"]["compiled"] is True


@pytest.mark.parametrize(
    ("method", "extra", "expected", "stages"),
    [
        (
            "weighted",
            (),
            (9, 1, 0),
            ["weighted_score"],
        ),
        (
            "lex-continuity",
            (),
            (8, 0, 1),
            ["continuity", "similarity", "overtime"],
        ),
        (
            "lex-overtime",
            (),
            (9, 1, 0),
            ["overtime", "continuity", "similarity"],
        ),
        (
            "epsilon",
            ("--delta", "0.2"),
            (8, 0, 1),
            [
                "similarity_reference",
                "continuity",
                "overtime",
                "similarity_tiebreak",
            ],
        ),
    ],
)
def test_reference_backend_certifies_objective_policies(
    method: str,
    extra: tuple[str, ...],
    expected: tuple[int, int, int],
    stages: list[str],
) -> None:
    result = _run(TRADEOFF, method, *extra)
    metrics = result["metrics"]
    assert result["status"] == "OPTIMUM"
    assert result["backend"] == "reference-enumerator"
    assert result["formulation"] == "direct-schedule-enumeration"
    assert result["solver_calls"] == len(stages)
    assert [stage["name"] for stage in result["stages"]] == stages
    assert all(stage["relative_gap"] == 0 for stage in result["stages"])
    assert metrics["verified"] is True
    assert (
        metrics["similarity"],
        metrics["continuity"],
        metrics["overtime"],
    ) == expected


def test_epsilon_uses_an_exact_decimal_ceiling() -> None:
    result = _run(TRADEOFF, "epsilon", "--delta", "0.2")
    assert result["similarity_reference_optimum"] == 9
    assert result["similarity_lower_bound"] == 8
    assert result["metrics"]["similarity"] >= 8


def test_reference_backend_accepts_omitted_empty_user_rows() -> None:
    result = _run(SPARSE_USERS, "weighted")
    assert result["status"] == "OPTIMUM"
    assert result["metrics"]["verified"] is True
    assert result["metrics"]["coverage"] == 2


def test_lex_cos_prioritizes_overtime_before_similarity() -> None:
    result = _run(LEX_COS_TIE, "lex-cos")
    assert result["status"] == "OPTIMUM"
    assert result["objective_policy"] == "continuity-overtime-similarity"
    assert [stage["name"] for stage in result["stages"]] == [
        "continuity",
        "overtime",
        "similarity",
    ]
    metrics = result["metrics"]
    assert (metrics["similarity"], metrics["continuity"], metrics["overtime"]) == (
        8,
        0,
        0,
    )


def test_soft_coverage_is_optimized_and_fixed_before_weighted_score() -> None:
    result = _run(PARTIAL, "weighted", "--soft-coverage")
    assert result["status"] == "OPTIMUM"
    assert [stage["name"] for stage in result["stages"]] == [
        "coverage",
        "weighted_score",
    ]
    assert result["stages"][0]["incumbent"] == 1
    assert result["metrics"]["coverage"] == 1
    assert result["metrics"]["verified"] is True


def test_full_coverage_reports_infeasible_when_a_service_has_no_candidate() -> None:
    _require_binary()
    completed = subprocess.run(
        [
            str(BINARY),
            str(PARTIAL),
            "--backend",
            "reference-enumerator",
            "--method",
            "weighted",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert result["status"] == "INFEASIBLE"
    assert result["metrics"] is None
    assert result["stages"][0]["status"] == "INFEASIBLE"


def test_certified_runs_reject_nonzero_optimality_gaps() -> None:
    _require_binary()
    completed = subprocess.run(
        [
            str(BINARY),
            str(TRADEOFF),
            "--backend",
            "reference-enumerator",
            "--method",
            "lex-continuity",
            "--mip-gap",
            "0.01",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    assert "require zero MIP gaps" in completed.stderr


@pytest.mark.parametrize("delta", (".", "1.1", "-0.1", "0.1234567890"))
def test_invalid_epsilon_budget_is_rejected(delta: str) -> None:
    _require_binary()
    completed = subprocess.run(
        [
            str(BINARY),
            str(TRADEOFF),
            "--backend",
            "reference-enumerator",
            "--method",
            "epsilon",
            "--delta",
            delta,
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    assert "delta" in completed.stderr


def test_reference_limit_returns_a_verified_timeout_incumbent() -> None:
    _require_binary()
    completed = subprocess.run(
        [
            str(BINARY),
            str(TRADEOFF),
            "--backend",
            "reference-enumerator",
            "--method",
            "weighted",
            "--enumeration-limit",
            "1",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert result["status"] == "TIMEOUT_FEASIBLE"
    assert result["metrics"]["verified"] is True
    assert result["incumbent_stage_index"] == 0
    assert result["stages"][0]["message"] == (
        "reference enumeration limit reached"
    )


def test_cumulative_timeout_can_expire_before_a_solver_call() -> None:
    _require_binary()
    completed = subprocess.run(
        [
            str(BINARY),
            str(TRADEOFF),
            "--backend",
            "reference-enumerator",
            "--method",
            "weighted",
            "--timeout",
            "0.000000000001",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    result = json.loads(completed.stdout)
    assert result["status"] == "TIMEOUT"
    assert result["solver_calls"] == 0
    assert result["metrics"] is None
    assert result["stages"][0]["message"] == (
        "cumulative timeout exhausted before stage construction"
    )
