"""Strict reader/writer for the text format consumed by the authors' C++ code."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .model import HCORAPInstance


class InstanceFormatError(ValueError):
    """Raised when an HCORAP input is malformed or internally inconsistent."""


SCALAR_TAGS = ("#U", "#S", "#A", "#TS", "#P")
SECTION_TAGS = (
    "#SU",
    "#SEQ",
    "#TSA(i)",
    "#TSS(i)",
    "#r(i,j)",
    "#HN(i)",
    "#HE(i)",
)
ALL_TAGS = set(SCALAR_TAGS) | set(SECTION_TAGS)


def _split_sections(text: str) -> Dict[str, List[str]]:
    sections: MutableMapping[str, List[str]] = {}
    current = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line not in ALL_TAGS:
                raise InstanceFormatError(
                    f"unknown tag {line!r} at line {line_number}"
                )
            if line in sections:
                raise InstanceFormatError(f"duplicate tag {line!r}")
            sections[line] = []
            current = line
            continue
        if current is None:
            raise InstanceFormatError(
                f"data appears before the first tag at line {line_number}"
            )
        sections[current].append(line)

    missing = ALL_TAGS - set(sections)
    if missing:
        raise InstanceFormatError(f"missing sections: {sorted(missing)}")
    return dict(sections)


def _scalar(sections: Mapping[str, Sequence[str]], tag: str) -> int:
    values = sections[tag]
    if len(values) != 1:
        raise InstanceFormatError(f"{tag} must contain exactly one integer")
    try:
        return int(values[0])
    except ValueError as exc:
        raise InstanceFormatError(f"invalid integer in {tag}: {values[0]!r}") from exc


def _groups(lines: Sequence[str], tag: str) -> List[List[int]]:
    groups = []
    for line in lines:
        try:
            group = [int(token) for token in line.split()]
        except ValueError as exc:
            raise InstanceFormatError(f"non-integer token in {tag}: {line!r}") from exc
        if not group:
            raise InstanceFormatError(f"empty group in {tag}")
        groups.append(group)
    return groups


def _matrix_from_tokens(
    lines: Sequence[str], rows: int, columns: int, tag: str
) -> List[List[int]]:
    tokens = " ".join(lines).split()
    expected = rows * columns
    if len(tokens) != expected:
        raise InstanceFormatError(
            f"{tag} has {len(tokens)} entries; expected {rows}x{columns}={expected}"
        )
    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise InstanceFormatError(f"non-integer token in {tag}") from exc
    return [values[start : start + columns] for start in range(0, expected, columns)]


def _vector(lines: Sequence[str], length: int, tag: str) -> List[int]:
    tokens = " ".join(lines).split()
    if len(tokens) != length:
        raise InstanceFormatError(
            f"{tag} has {len(tokens)} entries; expected {length}"
        )
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise InstanceFormatError(f"non-integer token in {tag}") from exc


def _validate_partition(groups: Sequence[Sequence[int]], size: int, tag: str) -> None:
    flattened = [item for group in groups for item in group]
    invalid = sorted({item for item in flattened if not 0 <= item < size})
    if invalid:
        raise InstanceFormatError(f"{tag} contains invalid service ids: {invalid[:10]}")
    duplicates = sorted(
        item for item, count in Counter(flattened).items() if count > 1
    )
    if duplicates:
        raise InstanceFormatError(f"{tag} repeats service ids: {duplicates[:10]}")
    missing = sorted(set(range(size)) - set(flattened))
    if missing:
        raise InstanceFormatError(f"{tag} omits service ids: {missing[:10]}")


def read_instance(path: Path) -> HCORAPInstance:
    """Parse and validate an instance without silently repairing bad input."""

    path = Path(path)
    sections = _split_sections(path.read_text(encoding="utf-8"))
    users = _scalar(sections, "#U")
    services = _scalar(sections, "#S")
    agents = _scalar(sections, "#A")
    time_slots = _scalar(sections, "#TS")
    penalty = _scalar(sections, "#P")
    if min(users, services, agents, time_slots) <= 0:
        raise InstanceFormatError("U, S, A and TS must all be positive")

    services_by_user = _groups(sections["#SU"], "#SU")
    sequences = _groups(sections["#SEQ"], "#SEQ")
    if len(services_by_user) > users:
        raise InstanceFormatError("#SU has more non-empty groups than users")
    _validate_partition(services_by_user, services, "#SU")
    _validate_partition(sequences, services, "#SEQ")

    tsa = _matrix_from_tokens(sections["#TSA(i)"], agents, time_slots, "#TSA(i)")
    tss = _matrix_from_tokens(
        sections["#TSS(i)"], services, time_slots, "#TSS(i)"
    )
    rewards = _matrix_from_tokens(sections["#r(i,j)"], agents, services, "#r(i,j)")
    normal = _vector(sections["#HN(i)"], agents, "#HN(i)")
    extra = _vector(sections["#HE(i)"], agents, "#HE(i)")

    for tag, matrix in (("#TSA(i)", tsa), ("#TSS(i)", tss)):
        bad = sorted({value for row in matrix for value in row if value not in (0, 1)})
        if bad:
            raise InstanceFormatError(f"{tag} must be binary; found {bad[:10]}")
    if any(value < 0 for row in rewards for value in row):
        raise InstanceFormatError("rewards must be non-negative")
    if any(value < 0 for value in normal + extra):
        raise InstanceFormatError("HN and HE must be non-negative")

    return HCORAPInstance(
        users=users,
        services=services,
        agents=agents,
        time_slots=time_slots,
        services_by_user=tuple(tuple(group) for group in services_by_user),
        sequences=tuple(tuple(group) for group in sequences),
        agent_availability=tuple(tuple(row) for row in tsa),
        service_availability=tuple(tuple(row) for row in tss),
        rewards=tuple(tuple(row) for row in rewards),
        overtime_penalty=penalty,
        normal_hours=tuple(normal),
        extra_hours=tuple(extra),
        source=path.resolve(),
    )


def _write_matrix(lines: List[str], matrix: Iterable[Sequence[int]]) -> None:
    for row in matrix:
        lines.append(" ".join(str(value) for value in row))


def write_instance(instance: HCORAPInstance, path: Path) -> None:
    """Write a file that remains consumable by the original C++ parser."""

    lines = [
        "#U",
        str(instance.users),
        "#S",
        str(instance.services),
        "#A",
        str(instance.agents),
        "#TS",
        str(instance.time_slots),
        "#SU",
    ]
    _write_matrix(lines, instance.services_by_user)
    lines.append("#SEQ")
    _write_matrix(lines, instance.sequences)
    lines.append("#TSA(i)")
    _write_matrix(lines, instance.agent_availability)
    lines.append("#TSS(i)")
    _write_matrix(lines, instance.service_availability)
    lines.append("#r(i,j)")
    _write_matrix(lines, instance.rewards)
    lines.extend(["#P", str(instance.overtime_penalty), "#HN(i)"])
    lines.extend(str(value) for value in instance.normal_hours)
    lines.append("#HE(i)")
    lines.extend(str(value) for value in instance.extra_hours)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
