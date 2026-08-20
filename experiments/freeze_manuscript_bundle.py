#!/usr/bin/env python3
"""Validate manuscript evidence and issue the LaTeX frozen-bundle marker.

The three generated manuscript fragments are deliberately insufficient on their
own.  ``main.tex`` accepts them only when this script has checked every enabled
analysis branch, cross-paradigm validation, screening decision, generator
provenance and source hashes, source commit, and the absence of draft tokens,
then generated ``freeze-manifest.tex``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


FRAGMENTS = (
    "abstract-findings.tex",
    "results.tex",
    "conclusion.tex",
)
FORBIDDEN_TOKENS = (
    r"\resultplaceholder",
    r"\outlineblock",
    "[citation pending]",
    "EXPLORATORY",
    "Replace this file",
    "Replace this conclusion",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def validate_and_render_marker(
    *,
    generated_dir: Path,
    primary_validation: Path,
    cross_validation: Path,
    corrected_validation: Path,
    generation_provenance: Path,
    screening_decision: Path,
    source_commit: str,
    expected_commit: str,
    source_clean: bool,
) -> str:
    primary = _json(primary_validation)
    cross = _json(cross_validation)
    screening = _json(screening_decision)
    provenance = _json(generation_provenance)
    errors: list[str] = []
    branches = screening.get("branches")
    if not isinstance(branches, dict):
        errors.append(f"screening decision has no branch map: {screening_decision}")
        branches = {}
    original_branch = branches.get("original_lexicographic", {})
    corrected_branch = branches.get("corrected_v2_lexicographic", {})
    original_enabled = original_branch.get("enabled") is True
    corrected_enabled = corrected_branch.get("enabled") is True

    if primary.get("valid") is not True:
        errors.append(f"primary analysis is not valid: {primary_validation}")
    if cross.get("valid") is not True:
        errors.append(f"cross-paradigm analysis is not valid: {cross_validation}")
    if screening.get("decision") != "GO":
        errors.append(f"screening decision is not GO: {screening_decision}")
    if provenance.get("valid") is not True:
        errors.append(f"manuscript generation provenance is invalid: {generation_provenance}")
    if not source_clean:
        errors.append("source worktree is not clean")
    if not expected_commit:
        errors.append("expected publication commit is empty")
    if source_commit != expected_commit:
        errors.append(
            f"source commit {source_commit} does not match expected {expected_commit}"
        )

    expected_primary_scope = "compact"
    if primary.get("scope") != expected_primary_scope:
        errors.append(
            "primary-analysis scope does not match screening branch: "
            f"{primary.get('scope')!r}/{expected_primary_scope!r}"
        )
    expected_cross_scope = "full"
    if cross.get("scope") != expected_cross_scope:
        errors.append(
            "cross-paradigm scope does not match screening branch: "
            f"{cross.get('scope')!r}/{expected_cross_scope!r}"
        )

    corrected: dict[str, Any] | None = None
    if corrected_enabled:
        try:
            corrected = _json(corrected_validation)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot read enabled corrected-v2 analysis: {error}")
        else:
            if corrected.get("valid") is not True:
                errors.append(
                    f"corrected-v2 analysis is not valid: {corrected_validation}"
                )

    if not original_enabled or not corrected_enabled:
        errors.append("compact scope requires both policy-validation branches")
    expected_measured = 1270
    if screening.get("expected_measured_runs") != expected_measured:
        errors.append(
            "screening measured-run total is inconsistent with enabled branches: "
            f"{screening.get('expected_measured_runs')!r}/{expected_measured}"
        )

    fragment_paths = [generated_dir / name for name in FRAGMENTS]
    for path in fragment_paths:
        if not path.is_file():
            errors.append(f"missing generated manuscript fragment: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                errors.append(f"draft token {token!r} remains in {path}")

    source_hashes = provenance.get("source_sha256")
    provenance_sources: list[Path] = []
    if not isinstance(source_hashes, dict) or not source_hashes:
        errors.append("manuscript provenance has no source-file hashes")
    else:
        for value, expected_hash in source_hashes.items():
            path = Path(value)
            provenance_sources.append(path)
            if not path.is_file():
                errors.append(f"manuscript provenance source is missing: {path}")
            elif _file_sha256(path) != expected_hash:
                errors.append(f"manuscript provenance source hash changed: {path}")

    fragment_hashes = provenance.get("fragment_sha256")
    if not isinstance(fragment_hashes, dict):
        errors.append("manuscript provenance has no fragment hashes")
    else:
        for path in fragment_paths:
            if path.is_file() and fragment_hashes.get(path.name) != _file_sha256(path):
                errors.append(f"generated fragment hash changed after rendering: {path}")

    provenance_expectations = {
        "primary_scope": primary.get("scope"),
        "cross_scope": cross.get("scope"),
        "original_lexicographic_enabled": original_enabled,
        "corrected_v2_lexicographic_enabled": corrected_enabled,
        "expected_measured_runs": screening.get("expected_measured_runs"),
    }
    for key, expected in provenance_expectations.items():
        if provenance.get(key) != expected:
            errors.append(
                f"manuscript provenance field {key} is stale: "
                f"{provenance.get(key)!r}/{expected!r}"
            )

    evidence_paths = [
        primary_validation,
        cross_validation,
        generation_provenance,
        screening_decision,
        *fragment_paths,
        *provenance_sources,
    ]
    if corrected_enabled and corrected is not None:
        evidence_paths.append(corrected_validation)
    if errors:
        raise ValueError("; ".join(errors))
    digest = _sha256(evidence_paths)
    scope = str(primary.get("scope", "unknown"))
    measured = screening["expected_measured_runs"]
    corrected_scope = "enabled" if corrected_enabled else "disabled"
    return (
        "% Generated by experiments/freeze_manuscript_bundle.py.\n"
        "% Do not edit: regenerate after any evidence or manuscript change.\n"
        r"\def\HCORAPFrozenValidationStatus{VALID}" "\n"
        rf"\def\HCORAPFrozenEvidenceDigest{{{digest}}}" "\n"
        rf"\def\HCORAPFrozenSourceCommit{{{_tex_escape(source_commit)}}}" "\n"
        rf"\def\HCORAPFrozenAnalysisScope{{{_tex_escape(scope)}}}" "\n"
        rf"\def\HCORAPFrozenCorrectedScope{{{corrected_scope}}}" "\n"
        rf"\def\HCORAPFrozenMeasuredRuns{{{_tex_escape(str(measured))}}}" "\n"
    )


def _git(command: list[str]) -> str:
    return subprocess.run(
        ["git", *command],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("LaTeX-Templates/paper/generated"),
    )
    parser.add_argument(
        "--primary-validation",
        type=Path,
        default=Path(
            "experiments/results/gcp_primary_analysis/analysis_validation.json"
        ),
    )
    parser.add_argument(
        "--cross-validation",
        type=Path,
        default=Path(
            "experiments/results/gcp_cross_paradigm_analysis/"
            "cross_paradigm_validation.json"
        ),
    )
    parser.add_argument(
        "--corrected-validation",
        type=Path,
        default=Path(
            "experiments/results/gcp_corrected_analysis/"
            "corrected_validation.json"
        ),
    )
    parser.add_argument(
        "--generation-provenance",
        type=Path,
        default=Path("LaTeX-Templates/paper/generated/manuscript-provenance.json"),
    )
    parser.add_argument(
        "--screening-decision",
        type=Path,
        default=Path("experiments/results/screening_decision.json"),
    )
    parser.add_argument(
        "--expected-commit",
        default=os.environ.get("HCORAP_EXPECTED_COMMIT", ""),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    marker = arguments.generated_dir / "freeze-manifest.tex"
    try:
        source_commit = _git(["rev-parse", "HEAD"])
        source_clean = not bool(_git(["status", "--porcelain"]))
        expected_commit = _git(
            ["rev-parse", "--verify", f"{arguments.expected_commit}^{{commit}}"]
        ) if arguments.expected_commit else ""
        content = validate_and_render_marker(
            generated_dir=arguments.generated_dir,
            primary_validation=arguments.primary_validation,
            cross_validation=arguments.cross_validation,
            corrected_validation=arguments.corrected_validation,
            generation_provenance=arguments.generation_provenance,
            screening_decision=arguments.screening_decision,
            source_commit=source_commit,
            expected_commit=expected_commit,
            source_clean=source_clean,
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        marker.unlink(missing_ok=True)
        raise SystemExit(f"refusing to freeze manuscript bundle: {error}") from error
    marker.write_text(content, encoding="utf-8")
    print(f"Frozen manuscript marker: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
