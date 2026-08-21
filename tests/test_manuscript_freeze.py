from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from experiments.freeze_manuscript_bundle import (
    FRAGMENTS,
    validate_and_render_marker,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    generated = tmp_path / "generated"
    generated.mkdir()
    for name in FRAGMENTS:
        (generated / name).write_text("Final evidence.\n", encoding="utf-8")
    primary = tmp_path / "primary.json"
    cross = tmp_path / "cross.json"
    corrected = tmp_path / "corrected.json"
    screen = tmp_path / "screen.json"
    primary.write_text(json.dumps({"valid": True, "scope": "compact"}), encoding="utf-8")
    cross.write_text(
        json.dumps({"valid": True, "scope": "full"}), encoding="utf-8"
    )
    corrected.write_text(
        json.dumps({"valid": True, "manuscript_eligible": True}),
        encoding="utf-8",
    )
    screen.write_text(
        json.dumps(
            {
                "decision": "GO",
                "expected_measured_runs": 924,
                "branches": {
                    "original_lexicographic": {"enabled": True},
                    "corrected_v2_lexicographic": {"enabled": True},
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = generated / "manuscript-provenance.json"
    sources = (primary, cross, corrected, screen)
    provenance.write_text(
        json.dumps(
            {
                "valid": True,
                "primary_scope": "compact",
                "cross_scope": "full",
                "original_lexicographic_enabled": True,
                "corrected_v2_lexicographic_enabled": True,
                "expected_measured_runs": 924,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources
                },
                "fragment_sha256": {
                    name: hashlib.sha256((generated / name).read_bytes()).hexdigest()
                    for name in FRAGMENTS
                },
            }
        ),
        encoding="utf-8",
    )
    return generated, primary, cross, corrected, screen, provenance


def test_valid_evidence_emits_valid_marker(tmp_path: Path) -> None:
    generated, primary, cross, corrected, screen, provenance = _inputs(tmp_path)
    marker = validate_and_render_marker(
        generated_dir=generated,
        primary_validation=primary,
        cross_validation=cross,
        corrected_validation=corrected,
        generation_provenance=provenance,
        screening_decision=screen,
        source_commit="abc123",
        expected_commit="abc123",
        source_clean=True,
    )
    assert r"\def\HCORAPFrozenValidationStatus{VALID}" in marker
    assert r"\def\HCORAPFrozenAnalysisScope{compact}" in marker
    assert r"\def\HCORAPFrozenMeasuredRuns{924}" in marker


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("invalid-primary", "primary analysis is not valid"),
        ("dirty", "worktree is not clean"),
        ("wrong-commit", "does not match expected"),
        ("draft-token", "draft token"),
        ("invalid-corrected", "not manuscript-eligible"),
    ],
)
def test_freeze_rejects_unvalidated_or_draft_inputs(
    tmp_path: Path, mutation: str, message: str
) -> None:
    generated, primary, cross, corrected, screen, provenance = _inputs(tmp_path)
    source_clean = True
    expected_commit = "abc123"
    if mutation == "invalid-primary":
        primary.write_text(json.dumps({"valid": False}), encoding="utf-8")
    elif mutation == "dirty":
        source_clean = False
    elif mutation == "wrong-commit":
        expected_commit = "different"
    elif mutation == "draft-token":
        (generated / "results.tex").write_text(
            r"\resultplaceholder{n}", encoding="utf-8"
        )
    elif mutation == "invalid-corrected":
        corrected.write_text(
            json.dumps({"valid": False, "manuscript_eligible": False}),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match=message):
        validate_and_render_marker(
            generated_dir=generated,
            primary_validation=primary,
            cross_validation=cross,
            corrected_validation=corrected,
            generation_provenance=provenance,
            screening_decision=screen,
            source_commit="abc123",
            expected_commit=expected_commit,
            source_clean=source_clean,
        )


def test_freeze_rejects_disabled_compact_branches(tmp_path: Path) -> None:
    generated, primary, cross, corrected, screen, provenance = _inputs(tmp_path)
    primary.write_text(
        json.dumps({"valid": True, "scope": "compact"}), encoding="utf-8"
    )
    cross.write_text(
        json.dumps({"valid": True, "scope": "full"}),
        encoding="utf-8",
    )
    corrected.unlink()
    screen.write_text(
        json.dumps(
            {
                "decision": "GO",
                "expected_measured_runs": 924,
                "branches": {
                    "original_lexicographic": {"enabled": False},
                    "corrected_v2_lexicographic": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    sources = (primary, cross, screen)
    provenance.write_text(
        json.dumps(
            {
                "valid": True,
                "primary_scope": "compact",
                "cross_scope": "full",
                "original_lexicographic_enabled": False,
                "corrected_v2_lexicographic_enabled": False,
                "expected_measured_runs": 924,
                "source_sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources
                },
                "fragment_sha256": {
                    name: hashlib.sha256((generated / name).read_bytes()).hexdigest()
                    for name in FRAGMENTS
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires both policy-validation branches"):
        validate_and_render_marker(
            generated_dir=generated,
            primary_validation=primary,
            cross_validation=cross,
            corrected_validation=corrected,
            generation_provenance=provenance,
            screening_decision=screen,
            source_commit="abc123",
            expected_commit="abc123",
            source_clean=True,
        )
