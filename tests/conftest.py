from __future__ import annotations

import pytest

from hcorap.model import HCORAPInstance


@pytest.fixture
def tradeoff_instance() -> HCORAPInstance:
    """Two services with a genuine SIM/CONT/OT trade-off."""

    return HCORAPInstance(
        users=1,
        services=2,
        agents=2,
        time_slots=2,
        services_by_user=((0, 1),),
        sequences=((0, 1),),
        agent_availability=((1, 1), (1, 1)),
        service_availability=((1, 1), (1, 1)),
        rewards=((5, 1), (4, 4)),
        overtime_penalty=-1,
        normal_hours=(1, 1),
        extra_hours=(1, 1),
    )

@pytest.fixture
def partially_infeasible_instance() -> HCORAPInstance:
    return HCORAPInstance(
        users=1,
        services=2,
        agents=2,
        time_slots=2,
        services_by_user=((0, 1),),
        sequences=((0, 1),),
        agent_availability=((1, 1), (1, 1)),
        service_availability=((1, 1), (0, 0)),
        rewards=((4, 4), (3, 3)),
        overtime_penalty=-1,
        normal_hours=(1, 1),
        extra_hours=(0, 0),
    )
