from __future__ import annotations

import json
from pathlib import Path

from experiments.audit_publication_evidence import _lex_transfer_check


def _write_decision(path: Path, decision: str) -> None:
    path.write_text(
        json.dumps(
            {
                "scope": "corrected-v2-lex-encoding-transfer",
                "structurally_valid": True,
                "decision": decision,
                "runs": 32,
                "instances": 16,
                "totalizer_only_optima": 0,
                "full_optima": 0,
                "stage_wins": 0,
                "gates": {
                    "at_least_two_net_extra_optima": False,
                    "extra_completed_stage_on_four_pairs": False,
                    "par2_improvement_at_least_ten_percent": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_lex_transfer_stop_is_accepted(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    _write_decision(decision, "STOP")

    report = _lex_transfer_check(decision)

    assert report["pass"]
    assert report["decision"] == "STOP"
    assert all(report["checks"].values())


def test_lex_transfer_go_is_rejected_without_confirmation(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    _write_decision(decision, "GO")

    report = _lex_transfer_check(decision)

    assert not report["pass"]
    assert not report["checks"]["decision_stop"]
