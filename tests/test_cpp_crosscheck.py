from __future__ import annotations

from pathlib import Path

import pytest

from hcorap.crosscheck import crosscheck_cpp_instance
from hcorap.io import write_instance


def test_cpp_weighted_objective_matches_proposed(
    tradeoff_instance, tmp_path: Path
) -> None:
    binary = Path("bin/release/hcorap2sat")
    if not binary.is_file():
        pytest.skip("authors' C++ encoder has not been built")
    instance_path = tmp_path / "tiny.txt"
    write_instance(tradeoff_instance, instance_path)
    result = crosscheck_cpp_instance(
        instance_path, binary=binary, timeout_seconds=10
    )
    assert result["status"] == "OPTIMUM"
    assert result["match"] is True
    assert result["cpp"]["equivalent_score"] == 8
    assert result["proposed"]["score"] == 8
