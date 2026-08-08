"""Corrected, seeded and auditable HCORAP benchmark generator."""

from __future__ import annotations

import json
import csv
import hashlib
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .io import read_instance, write_instance
from .metrics import verify_assignments
from .model import Assignment, HCORAPInstance


QUALIFICATIONS = ("basic", "nurse", "cpr", "physio", "doctor")
SERVICE_TYPES = ("basic", "basic", "basic", "nurse", "cpr", "physio", "doctor")
LANGUAGES = ("spanish", "catalan", "english")
GENDERS = ("M", "F")
RACES = ("white", "latin-american", "black", "asian")
LOAD_PROFILES = {
    "relaxed": 0.55,
    "critical": 0.85,
    "saturated": 0.98,
}


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
    witness_agent_limit: int | None = None

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
        if self.witness_agent_limit is not None and not (
            2 <= self.witness_agent_limit <= self.agents
        ):
            raise ValueError("witness_agent_limit must lie between 2 and agents")


def _sample_labels(
    rng: random.Random, labels: Sequence[str], minimum: int, maximum: int
) -> List[str]:
    count = rng.randint(minimum, min(maximum, len(labels)))
    return rng.sample(list(labels), count)


def _agent_attributes(config: GeneratorConfig, rng: random.Random) -> List[dict]:
    agents = []
    witness_agents = config.witness_agent_limit or config.agents
    for agent in range(config.agents):
        qualifications = _sample_labels(rng, QUALIFICATIONS, 1, 3)
        # The first two agents form a deterministic qualification backbone.
        # This preserves candidate availability when nested A subsets are used.
        if agent == 0:
            qualifications = ["basic", "nurse", "cpr"]
        elif agent == 1:
            qualifications = ["physio", "doctor", "basic"]
        elif agent < witness_agents:
            # Deterministic redundancy within every nested projection avoids
            # a single qualified caregiver becoming a feasibility bottleneck.
            required = QUALIFICATIONS[(agent - 2) % len(QUALIFICATIONS)]
            qualifications = sorted(set([*qualifications, required]))

        minimum_daily_hours = (
            min(config.slots_per_day, max(4, math.ceil(config.slots_per_day / 2)))
            if agent < witness_agents
            else min(4, config.slots_per_day)
        )
        daily_hours = rng.randint(minimum_daily_hours, config.slots_per_day)
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
    witness_agents = config.witness_agent_limit or config.agents
    occupied_agent_slots: set[tuple[int, int]] = set()
    occupied_user_slots: set[tuple[int, int]] = set()
    witness_workload = [0] * witness_agents
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

            candidates = []
            for agent in agents[:witness_agents]:
                if service_type not in agent["qualifications"]:
                    continue
                for slot, agent_available in enumerate(agent["availability"]):
                    if not agent_available:
                        continue
                    if (int(agent["id"]), slot) in occupied_agent_slots:
                        continue
                    if (user, slot) in occupied_user_slots:
                        continue
                    candidates.append(
                        (
                            0 if availability[slot] else 1,
                            witness_workload[int(agent["id"])],
                            int(agent["id"]),
                            slot,
                        )
                    )
            if not candidates:
                raise ValueError(
                    "cannot construct a full-coverage witness; increase the "
                    "minimum agent count or time horizon"
                )
            _repair, _workload, witness_agent, witness_slot = min(candidates)
            repaired_slot = None
            if not availability[witness_slot]:
                repaired_slot = witness_slot
                availability[witness_slot] = 1
            occupied_agent_slots.add((witness_agent, witness_slot))
            occupied_user_slots.add((user, witness_slot))
            witness_workload[witness_agent] += 1

            services.append(
                {
                    "id": len(services),
                    "user": user,
                    "visit": visit,
                    "type": service_type,
                    "availability": availability,
                    "candidate_repair_slot": repaired_slot,
                    "witness_agent": witness_agent,
                    "witness_slot": witness_slot,
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
                "constructive_full_coverage_witness": True,
            },
            "reward_quantiles": list(thresholds),
            "candidate_repair_count": sum(
                service["candidate_repair_slot"] is not None for service in services
            ),
            "agents_raw": agents,
            "users_raw": users,
            "services_raw": services,
            "witness": [
                {
                    "agent": service["witness_agent"],
                    "service": service["id"],
                    "time_slot": service["witness_slot"],
                }
                for service in services
            ],
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
    metadata["witness"] = [
        {
            "agent": service["witness_agent"],
            "service": service["id"],
            "time_slot": service["witness_slot"],
        }
        for service in selected_services
    ]

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
            witness_agent_limit=min(agent_values),
        )
    )
    return {
        (agents, visits): _project_master(
            master, agents=agents, services_per_user=visits
        )
        for agents in agent_values
        for visits in visit_values
    }


def _proportional_capacities(
    upper_bounds: Sequence[int], requested_total: int
) -> Tuple[int, ...]:
    """Allocate an integer total proportionally without exceeding availability."""

    available_total = sum(upper_bounds)
    target = min(max(0, requested_total), available_total)
    if available_total == 0:
        return tuple(0 for _ in upper_bounds)
    exact = [target * bound / available_total for bound in upper_bounds]
    allocated = [min(bound, math.floor(value)) for bound, value in zip(upper_bounds, exact)]
    remaining = target - sum(allocated)
    priority = sorted(
        range(len(upper_bounds)),
        key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
    )
    for index in priority:
        if remaining == 0:
            break
        if allocated[index] < upper_bounds[index]:
            allocated[index] += 1
            remaining -= 1
    if remaining:
        for index, bound in enumerate(upper_bounds):
            take = min(remaining, bound - allocated[index])
            allocated[index] += take
            remaining -= take
            if remaining == 0:
                break
    return tuple(allocated)


def calibrate_capacity(
    instance: HCORAPInstance,
    *,
    target_rho: float,
    normal_fraction: float = 0.85,
    load_profile: str | None = None,
) -> HCORAPInstance:
    """Calibrate usable capacity while preserving every scheduling domain.

    ``rho`` is services divided by total normal-plus-extra capacity.  Normal
    and overtime capacity are split deterministically after a proportional
    allocation bounded by each agent's available slots.
    """

    if not math.isfinite(target_rho) or target_rho <= 0:
        raise ValueError("target_rho must be finite and positive")
    if not math.isfinite(normal_fraction) or not 0 <= normal_fraction <= 1:
        raise ValueError("normal_fraction must lie in [0,1]")
    available = tuple(sum(row) for row in instance.agent_availability)
    requested_capacity = math.ceil(instance.services / target_rho)
    witness_load = [0] * instance.agents
    for assignment in instance.metadata.get("witness", ()):
        witness_load[int(assignment["agent"])] += 1
    if any(load > bound for load, bound in zip(witness_load, available)):
        raise ValueError("generation witness exceeds an agent availability bound")
    lower_total = sum(witness_load)
    target_total = min(
        max(requested_capacity, lower_total),
        sum(available),
    )
    residual = tuple(bound - lower for bound, lower in zip(available, witness_load))
    allocated_residual = _proportional_capacities(
        residual, target_total - lower_total
    )
    total_caps = tuple(
        lower + additional
        for lower, additional in zip(witness_load, allocated_residual)
    )
    normal_total = round(normal_fraction * sum(total_caps))
    normal = _proportional_capacities(total_caps, normal_total)
    extra = tuple(cap - regular for cap, regular in zip(total_caps, normal))
    realized_capacity = sum(total_caps)
    metadata = dict(instance.metadata)
    metadata["capacity_calibration"] = {
        "load_profile": load_profile,
        "target_rho": target_rho,
        "normal_fraction": normal_fraction,
        "requested_capacity": requested_capacity,
        "realized_capacity": realized_capacity,
        "realized_rho": (
            instance.services / realized_capacity if realized_capacity else None
        ),
        "availability_upper_bound": sum(available),
        "method": "deterministic-proportional-allocation",
        "witness_capacity_lower_bound": lower_total,
    }
    return replace(
        instance,
        normal_hours=normal,
        extra_hours=extra,
        metadata=metadata,
    )


def benchmark_diagnostics(instance: HCORAPInstance) -> Dict[str, object]:
    """Return pre-solve structural diagnostics used to stratify v2 runs."""

    summary = instance.to_summary()
    candidate_agents = []
    candidate_pairs = []
    service_windows = []
    for service in range(instance.services):
        pairs = instance.candidate_triplets(service)
        candidate_pairs.append(len(pairs))
        candidate_agents.append(len({agent for agent, _slot in pairs}))
        service_windows.append(sum(instance.service_availability[service]))
    qualified_pairs = sum(
        instance.rewards[agent][service] > 0
        for agent in range(instance.agents)
        for service in range(instance.services)
    )
    calibration = dict(instance.metadata.get("capacity_calibration", {}))
    return {
        **summary,
        "load_profile": calibration.get("load_profile"),
        "target_rho": calibration.get("target_rho"),
        "normal_capacity": sum(instance.normal_hours),
        "extra_capacity": sum(instance.extra_hours),
        "availability_upper_bound": sum(
            sum(row) for row in instance.agent_availability
        ),
        "candidate_agents_min": min(candidate_agents, default=0),
        "candidate_agents_mean": (
            sum(candidate_agents) / len(candidate_agents)
            if candidate_agents
            else 0.0
        ),
        "qualification_density": (
            qualified_pairs / (instance.agents * instance.services)
            if instance.agents and instance.services
            else 0.0
        ),
        "service_window_mean": (
            sum(service_windows) / len(service_windows)
            if service_windows
            else 0.0
        ),
        "candidate_pairs_max": max(candidate_pairs, default=0),
        "witness_assignments": len(instance.metadata.get("witness", ())),
    }


def generation_witness(instance: HCORAPInstance) -> Tuple[Assignment, ...]:
    """Materialize the auditable full-coverage witness stored by corrected-v2."""

    return tuple(
        Assignment(
            agent=int(item["agent"]),
            service=int(item["service"]),
            time_slot=int(item["time_slot"]),
        )
        for item in instance.metadata.get("witness", ())
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def generate_benchmark_batch(
    *,
    users: Iterable[int],
    agent_counts: Iterable[int],
    services_per_user_counts: Iterable[int],
    calibration_seeds: Iterable[int],
    evaluation_seeds: Iterable[int],
    load_profiles: Iterable[str],
    normal_fraction: float,
    output_dir: Path,
    days: int = 5,
    slots_per_day: int = 12,
    overtime_penalty: int = -1,
) -> Dict[str, object]:
    """Generate a frozen corrected-v2 batch with split and hash manifests."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = tuple(load_profiles)
    unknown = sorted(set(profiles) - set(LOAD_PROFILES))
    if unknown:
        raise ValueError(f"unknown load profiles: {unknown}")
    split_seeds = {
        "calibration": tuple(calibration_seeds),
        "evaluation": tuple(evaluation_seeds),
    }
    if not any(split_seeds.values()):
        raise ValueError("at least one calibration or evaluation seed is required")
    overlap = set(split_seeds["calibration"]) & set(split_seeds["evaluation"])
    if overlap:
        raise ValueError(f"calibration/evaluation seeds overlap: {sorted(overlap)}")

    rows: List[Dict[str, object]] = []
    for split, seeds in split_seeds.items():
        for seed in seeds:
            for user_count in sorted(set(users)):
                family = generate_nested_family(
                    users=user_count,
                    agent_counts=agent_counts,
                    services_per_user_counts=services_per_user_counts,
                    seed=seed,
                    days=days,
                    slots_per_day=slots_per_day,
                    overtime_penalty=overtime_penalty,
                )
                for profile in profiles:
                    for (agents, visits), base_instance in sorted(family.items()):
                        instance = calibrate_capacity(
                            base_instance,
                            target_rho=LOAD_PROFILES[profile],
                            normal_fraction=normal_fraction,
                            load_profile=profile,
                        )
                        name = (
                            f"instance_u{user_count}_a{agents}_v{visits}"
                            f"_seed{seed}_{profile}.txt"
                        )
                        text_path, metadata_path = write_generated_instance(
                            instance, output_dir / split / profile / name
                        )
                        serialized = read_instance(text_path)
                        witness_verified = verify_assignments(
                            serialized, generation_witness(instance)
                        ).valid
                        if not witness_verified:
                            raise RuntimeError(
                                f"serialized generation witness failed: {text_path}"
                            )
                        rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "users": user_count,
                                "agents": agents,
                                "visits": visits,
                                "load_profile": profile,
                                "instance": str(text_path.resolve()),
                                "metadata": str(metadata_path.resolve()),
                                "sha256": _sha256(text_path),
                                "witness_verified": witness_verified,
                                **benchmark_diagnostics(instance),
                            }
                        )

    diagnostics_path = output_dir / "diagnostics.csv"
    columns = list(rows[0]) if rows else []
    with diagnostics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "generator": "hcorap-corrected-v2",
        "load_profiles": {profile: LOAD_PROFILES[profile] for profile in profiles},
        "normal_fraction": normal_fraction,
        "calibration_seeds": list(split_seeds["calibration"]),
        "evaluation_seeds": list(split_seeds["evaluation"]),
        "instances": len(rows),
        "diagnostics": str(diagnostics_path.resolve()),
        "diagnostics_sha256": _sha256(diagnostics_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": str(manifest_path.resolve())}


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
