#!/usr/bin/env python3
"""Formulation-aware Git provenance checks for HCORAP experiments."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

# These files define the instance data consumed by both the MaxSAT and
# commercial formulations. Changes here invalidate cross-commit reuse.
SHARED_PROBLEM_SOURCE_PATHS = (
    "src/encodings/HCORAP/hcorap.cpp",
    "src/encodings/HCORAP/hcorap.h",
    "src/parser/parser.cpp",
    "src/parser/parser.h",
)

# These files define the solver-independent MIP-E formulation and commercial
# execution semantics shared by Gurobi and CPLEX.
COMMERCIAL_COMMON_SOURCE_PATHS = (
    "src/hcorap_commercial.cpp",
    "src/proposed/cpp/commercial/CommercialTypes.cpp",
    "src/proposed/cpp/commercial/CommercialTypes.h",
    "src/proposed/cpp/commercial/HCORAPMIPModel.cpp",
    "src/proposed/cpp/commercial/HCORAPMIPModel.h",
    *SHARED_PROBLEM_SOURCE_PATHS,
)

# A change to the Gurobi adapter or its build settings invalidates only the
# Gurobi evidence, not an independently measured CPLEX campaign.
GUROBI_SOURCE_PATHS = (
    "Makefile",
    "src/proposed/cpp/commercial/GurobiMIPBackend.cpp",
    *COMMERCIAL_COMMON_SOURCE_PATHS,
)


def git_paths_equivalent(
    left_commit: str,
    right_commit: str,
    paths: Iterable[str],
    *,
    root: Path = ROOT,
) -> bool:
    """Return whether selected tracked paths are identical at two commits."""
    if not left_commit or not right_commit:
        return False
    if left_commit == right_commit:
        return True
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            left_commit,
            right_commit,
            "--",
            *paths,
        ],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0
