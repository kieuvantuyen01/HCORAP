from __future__ import annotations

from hcorap.generator import LANGUAGES, generate_nested_family


def test_corrected_generator_is_seeded_exact_and_nested() -> None:
    first = generate_nested_family(
        users=3,
        agent_counts=(2, 4),
        services_per_user_counts=(2, 3),
        seed=19,
        days=2,
        slots_per_day=6,
    )
    second = generate_nested_family(
        users=3,
        agent_counts=(2, 4),
        services_per_user_counts=(2, 3),
        seed=19,
        days=2,
        slots_per_day=6,
    )
    assert first == second
    for (agents, visits), instance in first.items():
        assert instance.agents == agents
        assert instance.services == 3 * visits
        assert all(len(group) == visits for group in instance.services_by_user)
        assert instance.to_summary()["services_without_candidates"] == 0
        assert all(
            user["language"] in LANGUAGES
            for user in instance.metadata["users_raw"]
        )

    small = first[(2, 2)]
    large = first[(4, 3)]
    parent_ids = small.metadata["nested_parent"]["selected_parent_service_ids"]
    assert small.agent_availability == large.agent_availability[:2]
    assert small.service_availability == tuple(
        large.service_availability[parent] for parent in parent_ids
    )
