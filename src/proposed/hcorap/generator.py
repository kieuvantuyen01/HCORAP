"""Corrected, seeded and auditable HCORAP benchmark generator."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .io import write_instance
from .model import HCORAPInstance


QUALIFICATIONS = ("basic", "nurse", "cpr", "physio", "doctor")
SERVICE_TYPES = ("basic", "basic", "basic", "nurse", "cpr", "physio", "doctor")
LANGUAGES = ("spanish", "catalan", "english")
GENDERS = ("M", "F")
RACES = ("white", "latin-american", "black", "asian")


@dataclass(frozen=True)
class GeneratorConfig:
    users: int
    agents: int
    services_per_user: int
    seed: int
    days: int = 5
    slots_per_day: int = 12
    normal_hour_cap: int = 35
    overtime_penalty: int = -1

    @property
    def time_slots(self) -> int:
        return self.days * self.slots_per_day

    def validate(self) -> None:
        if min(self.users, self.agents, self.services_per_user) <= 0:
            raise ValueError("users, agents and services_per_user must be positive")
        if self.agents < 2:
            raise ValueError(
                "the corrected generator requires at least two agents so the "
                "complete qualification set can be represented"
            )
        if min(self.days, self.slots_per_day) <= 0:
            raise ValueError("days and slots_per_day must be positive")
        if self.normal_hour_cap < 0:
            raise ValueError("normal_hour_cap must be non-negative")


def _sample_labels(
    rng: random.Random, labels: Sequence[str], minimum: int, maximum: int
) -> List[str]:
    count = rng.randint(minimum, min(maximum, len(labels)))
    return rng.sample(list(labels), count)


def _agent_attributes(config: GeneratorConfig, rng: random.Random) -> List[dict]:
    agents = []
    for agent in range(config.agents):
        qualifications = _sample_labels(rng, QUALIFICATIONS, 1, 3)
        # The first two agents form a deterministic qualification backbone.
        # This preserves candidate availability when nested A subsets are used.
        if agent == 0:
            qualifications = ["basic", "nurse", "cpr"]
        elif agent == 1:
            qualifications = ["physio", "doctor", "basic"]

        daily_hours = rng.randint(4, config.slots_per_day)
        starts_early = rng.choice((True, False))
        if agent == 0:
            starts_early = True
        elif agent == 1:
            starts_early = False
        if starts_early:
            day = [1] * daily_hours + [0] * (config.slots_per_day - daily_hours)
        else:
            day = [0] * (config.slots_per_day - daily_hours) + [1] * daily_hours

        agents.append(
            {
                "id": agent,
                "age": rng.randint(25, 60),
                "qualifications": qualifications,
                "region_x": round(rng.random(), 6),
                "region_y": round(rng.random(), 6),
                "gender": rng.choice(GENDERS),
                "languages": _sample_labels(rng, LANGUAGES, 1, 2),
                "race": rng.choice(RACES),
                "availability": day * config.days,
            }
        )
    return agents


def _user_attributes(config: GeneratorConfig, rng: random.Random) -> List[dict]:
    users = []
    for user in range(config.users):
        users.append(
            {
                "id": user,
                "age": rng.randint(70, 90),
                "region_x": round(rng.random(), 6),
                "region_y": round(rng.random(), 6),
                "gender": rng.choice(GENDERS),
                # Corrected: sample from LANGUAGES, not from race labels.
                "language": rng.choice(LANGUAGES),
                "race": rng.choice(RACES),
            }
        )
    return users


def _service_attributes(
    config: GeneratorConfig,
    agents: Sequence[Mapping[str, object]],
    rng: random.Random,
) -> List[dict]:
    services = []
    for user in range(config.users):
        for visit in range(config.services_per_user):
            service_type = rng.choice(SERVICE_TYPES)
            day = (user + visit + rng.randrange(config.days)) % config.days
            window = rng.randint(2, min(5, config.slots_per_day))
            start = rng.randint(0, config.slots_per_day - window)
            availability = [0] * config.time_slots
            day_offset = day * config.slots_per_day
            for slot in range(start, start + window):
                availability[day_offset + slot] = 1

            qualified_backbone = next(
                agent
                for agent in agents[:2]
                if service_type in agent["qualifications"]
            )
            overlap = [
                slot
                for slot, value in enumerate(availability)
                if value and qualified_backbone["availability"][slot]
            ]
            repaired_slot = None
            if not overlap:
                same_day_slots = range(
                    day_offset, day_offset + config.slots_per_day
                )
                repaired_slot = next(
                    slot
                    for slot in same_day_slots
                    if qualified_backbone["availability"][slot]
                )
                availability[repaired_slot] = 1

            services.append(
                {
                    "id": len(services),
                    "user": user,
                    "visit": visit,
                    "type": service_type,
                    "availability": availability,
                    "candidate_repair_slot": repaired_slot,
                }
            )
    return services


def _raw_similarity(agent: Mapping[str, object], user: Mapping[str, object]) -> int:
    score = -int(user["age"] - agent["age"])
    if agent["gender"] == user["gender"]:
        score += 50
    if user["language"] in agent["languages"]:
        score += 100
    if agent["race"] == user["race"]:
        score += 25
    distance = math.hypot(
        float(agent["region_x"]) - float(user["region_x"]),
        float(agent["region_y"]) - float(user["region_y"]),
    )
    score -= int(50 * distance)
    return score


def _quantile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _reward_matrix(
    agents: Sequence[Mapping[str, object]],
    users: Sequence[Mapping[str, object]],
    services: Sequence[Mapping[str, object]],
) -> Tuple[List[List[int]], Tuple[float, float, float]]:
    raw: List[List[int | None]] = []
    valid_scores: List[int] = []
    for agent in agents:
        row = []
        for service in services:
            if service["type"] not in agent["qualifications"]:
                row.append(None)
                continue
            value = _raw_similarity(agent, users[int(service["user"])])
            row.append(value)
            valid_scores.append(value)
        raw.append(row)

    thresholds = (
        _quantile(valid_scores, 0.25),
        _quantile(valid_scores, 0.50),
        _quantile(valid_scores, 0.75),
    )
    rewards: List[List[int]] = []
    for row in raw:
        encoded = []
        for value in row:
            if value is None:
                encoded.append(0)
            elif value >= thresholds[2]:
                encoded.append(4)
            elif value >= thresholds[1]:
                encoded.append(3)
            elif value >= thresholds[0]:
                encoded.append(2)
            else:
                encoded.append(1)
        rewards.append(encoded)
    return rewards, thresholds


def _sequences(services: Sequence[Mapping[str, object]], users: int) -> List[List[int]]:
    result = []
    for user in range(users):
        for service_type in QUALIFICATIONS:
            group = [
                int(service["id"])
                for service in services
                if service["user"] == user and service["type"] == service_type
            ]
            if group:
                result.append(group)
    return result


def generate_instance(config: GeneratorConfig) -> HCORAPInstance:
    """Generate one deterministic corrected instance and retain audit metadata."""

    config.validate()
    rng = random.Random(config.seed)
    agents = _agent_attributes(config, rng)
    users = _user_attributes(config, rng)
    services = _service_attributes(config, agents, rng)
    rewards, thresholds = _reward_matrix(agents, users, services)

    normal_hours = []
    extra_hours = []
    for agent in agents:
        available = sum(int(value) for value in agent["availability"])
        normal_hours.append(min(available, config.normal_hour_cap))
        extra_hours.append(max(0, available - config.normal_hour_cap))

    instance = HCORAPInstance(
        users=config.users,
        services=config.users * config.services_per_user,
        agents=config.agents,
        time_slots=config.time_slots,
        services_by_user=tuple(
            tuple(
                int(service["id"])
                for service in services
                if service["user"] == user
            )
            for user in range(config.users)
        ),
        sequences=tuple(tuple(group) for group in _sequences(services, config.users)),
        agent_availability=tuple(
            tuple(int(value) for value in agent["availability"]) for agent in agents
        ),
        service_availability=tuple(
            tuple(int(value) for value in service["availability"])
            for service in services
        ),
        rewards=tuple(tuple(row) for row in rewards),
        overtime_penalty=config.overtime_penalty,
        normal_hours=tuple(normal_hours),
        extra_hours=tuple(extra_hours),
        metadata={
            "generator": "hcorap-corrected-v2",
            "seed": config.seed,
            "config": asdict(config),
            "corrections": {
                "exact_services_per_user": True,
                "language_domain": list(LANGUAGES),
                "nested_entity_order": True,
                "candidate_backbone_agents": [0, 1],
            },
            "reward_quantiles": list(thresholds),
            "candidate_repair_count": sum(
                service["candidate_repair_slot"] is not None for service in services
            ),
            "agents_raw": agents,
            "users_raw": users,
            "services_raw": services,
        },
    )
    return instance


def _project_master(
    master: HCORAPInstance, *, agents: int, services_per_user: int
) -> HCORAPInstance:
    raw_services = master.metadata["services_raw"]
    selected_old_ids = [
        int(service["id"])
        for service in raw_services
        if int(service["visit"]) < services_per_user
    ]
    remap = {old: new for new, old in enumerate(selected_old_ids)}
    selected_services = []
    for old in selected_old_ids:
        service = dict(raw_services[old])
        service["parent_id"] = old
        service["id"] = remap[old]
        selected_services.append(service)

    metadata = dict(master.metadata)
    metadata["config"] = dict(metadata["config"])
    metadata["config"]["agents"] = agents
    metadata["config"]["services_per_user"] = services_per_user
    metadata["nested_parent"] = {
        "agents": master.agents,
        "services_per_user": master.services // master.users,
        "selected_parent_service_ids": selected_old_ids,
    }
    metadata["agents_raw"] = list(master.metadata["agents_raw"][:agents])
    metadata["services_raw"] = selected_services
    metadata["candidate_repair_count"] = sum(
        service["candidate_repair_slot"] is not None
        for service in selected_services
    )

    return HCORAPInstance(
        users=master.users,
        services=master.users * services_per_user,
        agents=agents,
        time_slots=master.time_slots,
        services_by_user=tuple(
            tuple(
                int(service["id"])
                for service in selected_services
                if int(service["user"]) == user
            )
            for user in range(master.users)
        ),
        sequences=tuple(
            tuple(group) for group in _sequences(selected_services, master.users)
        ),
        agent_availability=master.agent_availability[:agents],
        service_availability=tuple(
            master.service_availability[old] for old in selected_old_ids
        ),
        rewards=tuple(
            tuple(master.rewards[agent][old] for old in selected_old_ids)
            for agent in range(agents)
        ),
        overtime_penalty=master.overtime_penalty,
        normal_hours=master.normal_hours[:agents],
        extra_hours=master.extra_hours[:agents],
        metadata=metadata,
    )


def generate_nested_family(
    *,
    users: int,
    agent_counts: Iterable[int],
    services_per_user_counts: Iterable[int],
    seed: int,
    days: int = 5,
    slots_per_day: int = 12,
    normal_hour_cap: int = 35,
    overtime_penalty: int = -1,
) -> Dict[Tuple[int, int], HCORAPInstance]:
    """Generate paired A/V variants by deterministic projection of one master."""

    agent_values = tuple(sorted(set(agent_counts)))
    visit_values = tuple(sorted(set(services_per_user_counts)))
    if not agent_values or not visit_values:
        raise ValueError("agent_counts and services_per_user_counts cannot be empty")
    if min(agent_values) < 2 or min(visit_values) <= 0:
        raise ValueError("all agent counts must be >=2 and visit counts must be >0")

    master = generate_instance(
        GeneratorConfig(
            users=users,
            agents=max(agent_values),
            services_per_user=max(visit_values),
            seed=seed,
            days=days,
            slots_per_day=slots_per_day,
            normal_hour_cap=normal_hour_cap,
            overtime_penalty=overtime_penalty,
        )
    )
    return {
        (agents, visits): _project_master(
            master, agents=agents, services_per_user=visits
        )
        for agents in agent_values
        for visits in visit_values
    }


def write_generated_instance(instance: HCORAPInstance, path: Path) -> Tuple[Path, Path]:
    """Write C++-compatible TXT plus an auditable JSON sidecar."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_instance(instance, path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    payload = {
        "instance": instance.to_summary(),
        "metadata": dict(instance.metadata),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path, metadata_path
