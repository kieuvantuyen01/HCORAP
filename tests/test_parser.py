from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hcorap.io import InstanceFormatError, read_instance, write_instance


OFFICIAL = Path(
    "instances/paperInstances/TXT_10-25_4-5_U30/instance_30_15_4_47.txt"
)


def test_parse_official_instance() -> None:
    instance = read_instance(OFFICIAL)
    assert (instance.users, instance.services, instance.agents) == (30, 120, 15)
    assert instance.time_slots == 60
    assert sorted(service for group in instance.services_by_user for service in group) == list(
        range(120)
    )
    assert sorted(service for group in instance.sequences for service in group) == list(
        range(120)
    )


def test_round_trip_is_lossless(tmp_path: Path) -> None:
    original = read_instance(OFFICIAL)
    target = tmp_path / "roundtrip.txt"
    write_instance(original, target)
    reread = read_instance(target)
    assert reread == replace(original, source=reread.source)


def test_parser_rejects_duplicate_service(tmp_path: Path) -> None:
    text = OFFICIAL.read_text(encoding="utf-8")
    text = text.replace("16 42 50 57", "16 16 50 57", 1)
    malformed = tmp_path / "duplicate.txt"
    malformed.write_text(text, encoding="utf-8")
    with pytest.raises(InstanceFormatError, match="repeats service"):
        read_instance(malformed)
