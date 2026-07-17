from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "bin" / "release" / "hcorap_multi"
INSTANCE = ROOT / "tests" / "instances" / "tradeoff.txt"
CARDINALITY_INSTANCE = ROOT / "tests" / "instances" / "cardinality.txt"
PARTIAL_INSTANCE = ROOT / "tests" / "instances" / "partial_coverage.txt"
SYMMETRY_INSTANCE = ROOT / "tests" / "instances" / "symmetry.txt"
SYMMETRY_PARTIAL_INSTANCE = (
    ROOT / "tests" / "instances" / "symmetry_partial.txt"
)
RC2_STUB = ROOT / "tests" / "rc2_open_wbo.py"


def _solver() -> str:
    configured = os.environ.get("OPEN_WBO_BIN")
    discovered = shutil.which("open-wbo")
    solver = configured or discovered or str(RC2_STUB)
    if not solver or not Path(solver).is_file():
        pytest.skip("set OPEN_WBO_BIN to run C++ end-to-end solver tests")
    return solver


def _run_instance(instance: Path, method: str, *extra: str) -> dict:
    if not BINARY.is_file():
        pytest.skip("build hcorap_multi with: make -j4 YICES=0")
    completed = subprocess.run(
        [
            str(BINARY),
            str(instance),
            "--solver",
            _solver(),
            "--timeout",
            "30",
            "--method",
            method,
            *extra,
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=35,
    )
    return json.loads(completed.stdout)


def _run(method: str, *extra: str) -> dict:
    return _run_instance(INSTANCE, method, *extra)


def test_cpp_encode_only_is_valid_legacy_wcnf(tmp_path: Path) -> None:
    if not BINARY.is_file():
        pytest.skip("build hcorap_multi with: make -j4 YICES=0")
    formula_path = tmp_path / "tradeoff.wcnf"
    with formula_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            [
                str(BINARY),
                str(INSTANCE),
                "--encode-only",
                "--cardinality-encoding",
                "sorting-network",
            ],
            check=True,
            text=True,
            stdout=output,
        )

    header = formula_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("p wcnf 22 83 ")
    formula = WCNF(from_file=str(formula_path))
    with RC2(formula) as solver:
        assert solver.compute() is not None
        assert solver.cost == 7


def test_cpp_totalizer_preserves_optimum_with_a_distinct_encoding() -> None:
    if not BINARY.is_file():
        pytest.skip("build hcorap_multi with: make -j4 YICES=0")

    formulas = {}
    costs = {}
    for encoding in ("sorting-network", "totalizer"):
        completed = subprocess.run(
            [
                str(BINARY),
                str(CARDINALITY_INSTANCE),
                "--encode-only",
                "--cardinality-encoding",
                encoding,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        formula = WCNF(from_string=completed.stdout)
        formulas[encoding] = (formula.nv, len(formula.hard))
        with RC2(formula) as solver:
            assert solver.compute() is not None
            costs[encoding] = solver.cost

    assert formulas["sorting-network"] != formulas["totalizer"]
    assert costs["sorting-network"] == costs["totalizer"]


@pytest.mark.parametrize(
    "implied_config",
    ("none", "user-slots", "slot-capacity", "both", "both-plus"),
)
@pytest.mark.parametrize("cardinality_encoding", ("sorting-network", "totalizer"))
def test_cpp_implied_configs_preserve_weighted_optimum(
    implied_config: str, cardinality_encoding: str
) -> None:
    completed = subprocess.run(
        [
            str(BINARY),
            str(CARDINALITY_INSTANCE),
            "--encode-only",
            "--cardinality-encoding",
            cardinality_encoding,
            "--implied-constraints",
            implied_config,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    formula = WCNF(from_string=completed.stdout)
    with RC2(formula) as solver:
        assert solver.compute() is not None
        assert solver.cost == 3


@pytest.mark.parametrize(
    ("method", "extra", "expected"),
    [
        ("weighted", (), (9, 1, 0)),
        ("lex-continuity", (), (8, 0, 1)),
        ("lex-overtime", (), (9, 1, 0)),
        ("epsilon", ("--delta", "0"), (9, 1, 0)),
        ("epsilon", ("--delta", "0.2"), (8, 0, 1)),
    ],
)
def test_cpp_methods_recover_expected_tradeoffs(
    method: str, extra: tuple[str, ...], expected: tuple[int, int, int]
) -> None:
    result = _run(method, *extra)
    metrics = result["metrics"]
    assert result["status"] == "OPTIMUM"
    assert result["language"] == "C++"
    assert result["cardinality_encoding"] == "sorting-network"
    assert result["implied_constraints"] == "none"
    assert result["symmetry_breaking"] == "none"
    assert metrics["verified"] is True
    assert (
        metrics["similarity"],
        metrics["continuity"],
        metrics["overtime"],
    ) == expected


def test_cpp_soft_coverage_is_optimized_before_quality() -> None:
    result = _run("weighted", "--soft-coverage")
    assert result["status"] == "OPTIMUM"
    assert [stage["objective"] for stage in result["stages"]] == [
        "coverage",
        "weighted_score",
    ]
    assert result["metrics"]["coverage"] == 2


def test_cpp_totalizer_result_is_labeled_and_verified() -> None:
    result = _run("weighted", "--cardinality-encoding", "totalizer")
    assert result["status"] == "OPTIMUM"
    assert result["cardinality_encoding"] == "totalizer"
    assert result["metrics"]["verified"] is True


@pytest.mark.parametrize(
    "implied_config",
    ("user-slots", "slot-capacity", "both", "both-plus"),
)
def test_cpp_implied_configs_are_labeled_and_verified(
    implied_config: str,
) -> None:
    result = _run(
        "weighted",
        "--implied-constraints",
        implied_config,
    )
    assert result["status"] == "OPTIMUM"
    assert result["implied_constraints"] == implied_config
    assert result["metrics"]["verified"] is True


@pytest.mark.parametrize(
    "symmetry_config",
    ("none", "slots", "services", "slot-service", "all"),
)
def test_cpp_symmetry_configs_preserve_optimum_and_verify(
    symmetry_config: str,
) -> None:
    result = _run_instance(
        SYMMETRY_INSTANCE,
        "weighted",
        "--symmetry-breaking",
        symmetry_config,
    )
    assert result["status"] == "OPTIMUM"
    assert result["symmetry_breaking"] == symmetry_config
    assert result["metrics"]["similarity"] == 4
    assert result["metrics"]["verified"] is True


def test_cpp_symmetry_detection_has_no_encoding_overhead_without_classes() -> None:
    formulas = []
    for symmetry_config in ("none", "all"):
        completed = subprocess.run(
            [
                str(BINARY),
                str(CARDINALITY_INSTANCE),
                "--encode-only",
                "--symmetry-breaking",
                symmetry_config,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        formulas.append(completed.stdout)
    assert formulas[0] == formulas[1]


@pytest.mark.parametrize(
    "symmetry_config",
    ("services", "slot-service", "all"),
)
def test_cpp_service_symmetry_supports_partial_coverage(
    symmetry_config: str,
) -> None:
    result = _run_instance(
        SYMMETRY_PARTIAL_INSTANCE,
        "weighted",
        "--soft-coverage",
        "--symmetry-breaking",
        symmetry_config,
    )
    assert result["status"] == "OPTIMUM"
    assert result["metrics"]["coverage"] == 1
    assert result["metrics"]["verified"] is True


@pytest.mark.parametrize(
    "symmetry_config",
    ("none", "slots", "services", "slot-service", "all"),
)
@pytest.mark.parametrize(
    "implied_config",
    ("none", "user-slots", "slot-capacity", "both", "both-plus"),
)
@pytest.mark.parametrize("cardinality_encoding", ("sorting-network", "totalizer"))
def test_cpp_full_configuration_matrix_preserves_optimum(
    symmetry_config: str,
    implied_config: str,
    cardinality_encoding: str,
) -> None:
    completed = subprocess.run(
        [
            str(BINARY),
            str(SYMMETRY_INSTANCE),
            "--encode-only",
            "--cardinality-encoding",
            cardinality_encoding,
            "--implied-constraints",
            implied_config,
            "--symmetry-breaking",
            symmetry_config,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    formula = WCNF(from_string=completed.stdout)
    with RC2(formula) as solver:
        assert solver.compute() is not None
        assert solver.cost == 6


@pytest.mark.parametrize(
    "implied_config",
    ("user-slots", "both", "both-plus"),
)
def test_cpp_user_slot_configs_support_soft_coverage(
    implied_config: str,
) -> None:
    result = _run(
        "weighted",
        "--soft-coverage",
        "--implied-constraints",
        implied_config,
    )
    assert result["status"] == "OPTIMUM"
    assert result["metrics"]["coverage"] == 2
    assert result["metrics"]["verified"] is True


@pytest.mark.parametrize(
    "implied_config",
    ("user-slots", "both", "both-plus"),
)
def test_cpp_user_slot_cardinality_tracks_partial_coverage(
    implied_config: str,
) -> None:
    result = _run_instance(
        PARTIAL_INSTANCE,
        "weighted",
        "--soft-coverage",
        "--implied-constraints",
        implied_config,
    )
    assert result["status"] == "OPTIMUM"
    assert result["metrics"]["coverage"] == 1
    assert result["metrics"]["verified"] is True


def test_cpp_timeout_is_cumulative_and_machine_readable() -> None:
    if not BINARY.is_file():
        pytest.skip("build hcorap_multi with: make -j4 YICES=0")
    completed = subprocess.run(
        [
            str(BINARY),
            str(INSTANCE),
            "--solver",
            _solver(),
            "--timeout",
            "0.001",
            "--method",
            "epsilon",
            "--delta",
            "0.2",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=5,
    )
    result = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert result["status"] == "TIMEOUT"
    assert result["metrics"] is None
    assert result["timing_scope"] == "parse+encode+serialize+solve+verify"


def test_cpp_rejects_unknown_implied_config() -> None:
    if not BINARY.is_file():
        pytest.skip("build hcorap_multi with: make -j4 YICES=0")
    completed = subprocess.run(
        [
            str(BINARY),
            str(INSTANCE),
            "--encode-only",
            "--implied-constraints",
            "unknown",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "unsupported implied-constraints configuration" in completed.stderr


def test_cpp_rejects_unknown_symmetry_config() -> None:
    if not BINARY.is_file():
        pytest.skip("build hcorap_multi with: make -j4 YICES=0")
    completed = subprocess.run(
        [
            str(BINARY),
            str(INSTANCE),
            "--encode-only",
            "--symmetry-breaking",
            "unknown",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "unsupported symmetry-breaking configuration" in completed.stderr
