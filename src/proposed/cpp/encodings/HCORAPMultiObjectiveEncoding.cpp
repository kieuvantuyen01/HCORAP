#include "HCORAPMultiObjectiveEncoding.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>

using namespace std;

HCORAPObjectiveBounds::HCORAPObjectiveBounds()
    : minCoverage(-1), minSimilarity(-1), maxContinuity(-1), maxOvertime(-1) {}

HCORAPSolutionMetrics::HCORAPSolutionMetrics()
    : valid(false), coverage(0), similarity(0), continuity(0), overtime(0),
      overtimeCost(0) {}

HCORAPMultiObjectiveEncoding::HCORAPMultiObjectiveEncoding(
    HCORAP *instance,
    HCORAPObjectiveKind objective,
    bool fullCoverage,
    int continuityWeight,
    int overtimeWeight,
    HCORAPCardinalityEncoding cardinalityEncoding,
    HCORAPImpliedConfig impliedConfig,
    HCORAPSymmetryBreaking symmetryBreaking,
    const HCORAPObjectiveBounds &bounds
) : instance(instance), objective(objective), bounds(bounds),
    fullCoverage(fullCoverage), continuityWeight(continuityWeight),
    overtimeWeight(overtimeWeight), cardinalityEncoding(cardinalityEncoding),
    impliedConfig(impliedConfig), symmetryBreaking(symmetryBreaking) {}

void HCORAPMultiObjectiveEncoding::addAtMostOne(
    SMTFormula *formula, const vector<literal> &values
) {
    if (values.size() > 1)
        formula->addAMO(values);
}

void HCORAPMultiObjectiveEncoding::addCardinalityAtMost(
    SMTFormula *formula,
    const vector<literal> &values,
    int bound
) {
    if (bound < 0) {
        formula->addClause(formula->falseVar());
        return;
    }
    if (static_cast<int>(values.size()) <= bound)
        return;
    if (bound == 0) {
        for (const literal &value : values)
            formula->addClause(!value);
        return;
    }
    vector<literal> thresholds;
    addHCORAPCardinalityNetwork(
        formula, values, thresholds, cardinalityEncoding
    );
    formula->addClause(!thresholds[bound]);
}

void HCORAPMultiObjectiveEncoding::addCardinalityExactly(
    SMTFormula *formula,
    const vector<literal> &values,
    int target
) {
    if (target < 0 || target > static_cast<int>(values.size())) {
        formula->addClause(formula->falseVar());
        return;
    }
    if (target == 0) {
        for (const literal &value : values)
            formula->addClause(!value);
        return;
    }
    if (target == static_cast<int>(values.size())) {
        for (const literal &value : values)
            formula->addClause(value);
        return;
    }
    vector<literal> thresholds;
    addHCORAPCardinalityNetwork(
        formula, values, thresholds, cardinalityEncoding
    );
    formula->addClause(thresholds[target - 1]);
    formula->addClause(!thresholds[target]);
}

void HCORAPMultiObjectiveEncoding::addEqualCardinality(
    SMTFormula *formula,
    const vector<literal> &left,
    const vector<literal> &right
) {
    vector<literal> leftThresholds;
    vector<literal> rightThresholds;
    addHCORAPCardinalityNetwork(
        formula, left, leftThresholds, cardinalityEncoding
    );
    addHCORAPCardinalityNetwork(
        formula, right, rightThresholds, cardinalityEncoding
    );

    const size_t shared = min(leftThresholds.size(), rightThresholds.size());
    for (size_t index = 0; index < shared; ++index) {
        formula->addClause(
            !leftThresholds[index] | rightThresholds[index]
        );
        formula->addClause(
            leftThresholds[index] | !rightThresholds[index]
        );
    }
    for (size_t index = shared; index < leftThresholds.size(); ++index)
        formula->addClause(!leftThresholds[index]);
    for (size_t index = shared; index < rightThresholds.size(); ++index)
        formula->addClause(!rightThresholds[index]);
}

void HCORAPMultiObjectiveEncoding::addServiceSlotVariables(
    SMTFormula *formula
) {
    serviceSlot = vector<vector<literal> >(
        instance->S, vector<literal>(instance->TS, formula->falseVar())
    );
    for (int service = 0; service < instance->S; ++service) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            vector<literal> inputs;
            for (int agent = 0; agent < instance->A; ++agent) {
                if (x[agent][service][slot].v.id != formula->falseVar().id)
                    inputs.push_back(x[agent][service][slot]);
            }
            if (inputs.empty())
                continue;
            literal output = formula->newBoolVar(
                "service_slot", service, slot
            );
            serviceSlot[service][slot] = output;
            clause reverse = !output;
            for (const literal &input : inputs) {
                formula->addClause(!input | output);
                reverse |= input;
            }
            formula->addClause(reverse);
        }
    }
}

void HCORAPMultiObjectiveEncoding::addProjectedServiceAssignments(
    SMTFormula *formula
) {
    for (int service = 0; service < instance->S; ++service) {
        vector<literal> candidates;
        clause reverse = !performed[service];
        for (int agent = 0; agent < instance->A; ++agent) {
            bool feasible = false;
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (x[agent][service][slot].v.id != formula->falseVar().id) {
                    feasible = true;
                    break;
                }
            }
            if (!feasible)
                continue;
            candidates.push_back(y[agent][service]);
            reverse |= y[agent][service];
            formula->addClause(
                !y[agent][service] | performed[service]
            );
        }
        formula->addClause(reverse);
        addAtMostOne(formula, candidates);
    }
}

void HCORAPMultiObjectiveEncoding::addUserSlotConstraints(
    SMTFormula *formula
) {
    userUsedSlot = vector<vector<literal> >(
        instance->SU.size(),
        vector<literal>(instance->TS, formula->falseVar())
    );
    for (int user = 0; user < static_cast<int>(instance->SU.size()); ++user) {
        vector<literal> usedSlots;
        for (int slot = 0; slot < instance->TS; ++slot) {
            vector<literal> inputs;
            for (int service : instance->SU[user]) {
                if (serviceSlot[service][slot].v.id != formula->falseVar().id)
                    inputs.push_back(serviceSlot[service][slot]);
            }
            if (inputs.empty())
                continue;
            literal output = formula->newBoolVar("user_slot", user, slot);
            userUsedSlot[user][slot] = output;
            usedSlots.push_back(output);
            clause reverse = !output;
            for (const literal &input : inputs) {
                formula->addClause(!input | output);
                reverse |= input;
            }
            formula->addClause(reverse);
        }

        if (fullCoverage) {
            addCardinalityExactly(
                formula,
                usedSlots,
                static_cast<int>(instance->SU[user].size())
            );
        } else {
            vector<literal> servedServices;
            for (int service : instance->SU[user])
                servedServices.push_back(performed[service]);
            addEqualCardinality(formula, usedSlots, servedServices);
        }
    }
}

int HCORAPMultiObjectiveEncoding::slotMatchingCapacity(int slot) const {
    vector<int> serviceUser(instance->S, -1);
    bool validPartition = true;
    for (int user = 0; user < static_cast<int>(instance->SU.size()); ++user) {
        for (int service : instance->SU[user]) {
            if (service < 0 || service >= instance->S ||
                serviceUser[service] != -1) {
                validPartition = false;
                continue;
            }
            serviceUser[service] = user;
        }
    }

    int feasibleServices = 0;
    for (int service = 0; service < instance->S; ++service) {
        bool feasible = false;
        for (int agent = 0; agent < instance->A; ++agent) {
            if (instance->r[agent][service] > 0 &&
                instance->TSA[agent][slot] &&
                instance->TSS[service][slot]) {
                feasible = true;
                break;
            }
        }
        if (feasible)
            ++feasibleServices;
        if (serviceUser[service] < 0)
            validPartition = false;
    }
    if (!validPartition)
        return feasibleServices;

    vector<vector<int> > adjacency(instance->A);
    for (int agent = 0; agent < instance->A; ++agent) {
        set<int> users;
        for (int service = 0; service < instance->S; ++service) {
            if (instance->r[agent][service] > 0 &&
                instance->TSA[agent][slot] &&
                instance->TSS[service][slot])
                users.insert(serviceUser[service]);
        }
        adjacency[agent].assign(users.begin(), users.end());
    }

    vector<int> matchedUser(instance->SU.size(), -1);
    int matching = 0;
    for (int agent = 0; agent < instance->A; ++agent) {
        vector<bool> seen(instance->SU.size(), false);
        function<bool(int)> augment = [&](int currentAgent) {
            for (int user : adjacency[currentAgent]) {
                if (seen[user])
                    continue;
                seen[user] = true;
                if (matchedUser[user] < 0 || augment(matchedUser[user])) {
                    matchedUser[user] = currentAgent;
                    return true;
                }
            }
            return false;
        };
        if (augment(agent))
            ++matching;
    }
    return matching;
}

void HCORAPMultiObjectiveEncoding::addSlotCapacityConstraints(
    SMTFormula *formula
) {
    for (int slot = 0; slot < instance->TS; ++slot) {
        vector<literal> scheduledServices;
        for (int service = 0; service < instance->S; ++service) {
            if (serviceSlot[service][slot].v.id != formula->falseVar().id)
                scheduledServices.push_back(serviceSlot[service][slot]);
        }
        addCardinalityAtMost(
            formula, scheduledServices, slotMatchingCapacity(slot)
        );
    }
}

void HCORAPMultiObjectiveEncoding::addValuePrecedence(
    SMTFormula *formula,
    const vector<literal> &earlier,
    const vector<literal> &later
) {
    int lastLater = -1;
    for (int position = 0; position < static_cast<int>(later.size()); ++position) {
        if (later[position].v.id != formula->falseVar().id)
            lastLater = position;
    }
    if (lastLater < 0)
        return;

    literal earlierPrefix = formula->falseVar();
    for (int position = 0; position <= lastLater; ++position) {
        if (later[position].v.id != formula->falseVar().id)
            formula->addClause(!later[position] | earlierPrefix);

        if (position == lastLater ||
            earlier[position].v.id == formula->falseVar().id)
            continue;
        if (earlierPrefix.v.id == formula->falseVar().id) {
            earlierPrefix = earlier[position];
            continue;
        }

        literal extendedPrefix = formula->newBoolVar();
        formula->addClause(!earlierPrefix | extendedPrefix);
        formula->addClause(!earlier[position] | extendedPrefix);
        formula->addClause(
            !extendedPrefix | earlierPrefix | earlier[position]
        );
        earlierPrefix = extendedPrefix;
    }
}

bool HCORAPMultiObjectiveEncoding::hasSlotSymmetry() const {
    set<vector<int> > signatures;
    for (int slot = 0; slot < instance->TS; ++slot) {
        vector<int> signature;
        bool active = false;
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int service = 0; service < instance->S; ++service) {
                int feasible = instance->r[agent][service] > 0 &&
                    instance->TSA[agent][slot] &&
                    instance->TSS[service][slot];
                signature.push_back(feasible);
                active = active || feasible;
            }
        }
        if (active && !signatures.insert(signature).second)
            return true;
    }
    return false;
}

void HCORAPMultiObjectiveEncoding::addSlotSymmetryBreaking(
    SMTFormula *formula
) {
    map<vector<int>, vector<int> > classes;
    for (int slot = 0; slot < instance->TS; ++slot) {
        vector<int> signature;
        bool active = false;
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int service = 0; service < instance->S; ++service) {
                int feasible = x[agent][service][slot].v.id !=
                    formula->falseVar().id;
                signature.push_back(feasible);
                active = active || feasible;
            }
        }
        if (active)
            classes[signature].push_back(slot);
    }

    for (const auto &entry : classes) {
        const vector<int> &slots = entry.second;
        for (size_t index = 1; index < slots.size(); ++index) {
            vector<literal> earlier(instance->S, formula->falseVar());
            vector<literal> later(instance->S, formula->falseVar());
            for (int service = 0; service < instance->S; ++service) {
                earlier[service] = serviceSlot[service][slots[index - 1]];
                later[service] = serviceSlot[service][slots[index]];
            }
            addValuePrecedence(formula, earlier, later);
        }
    }
}

void HCORAPMultiObjectiveEncoding::addServiceSymmetryBreaking(
    SMTFormula *formula
) {
    vector<int> serviceUser(instance->S, -1);
    vector<int> serviceSequence(instance->S, -1);
    for (int user = 0; user < static_cast<int>(instance->SU.size()); ++user) {
        for (int service : instance->SU[user]) {
            if (service < 0 || service >= instance->S)
                continue;
            serviceUser[service] = serviceUser[service] == -1 ? user : -2;
        }
    }
    for (int sequence = 0;
         sequence < static_cast<int>(instance->SEQ.size()); ++sequence) {
        for (int service : instance->SEQ[sequence]) {
            if (service < 0 || service >= instance->S)
                continue;
            serviceSequence[service] = serviceSequence[service] == -1
                ? sequence : -2;
        }
    }

    map<vector<int>, vector<int> > classes;
    for (int service = 0; service < instance->S; ++service) {
        if (serviceUser[service] < 0 || serviceSequence[service] < 0)
            continue;
        vector<int> signature;
        signature.push_back(serviceUser[service]);
        signature.push_back(serviceSequence[service]);
        for (int agent = 0; agent < instance->A; ++agent)
            signature.push_back(instance->r[agent][service]);
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                signature.push_back(
                    x[agent][service][slot].v.id != formula->falseVar().id
                );
            }
        }
        classes[signature].push_back(service);
    }

    const int positions = instance->A * instance->TS;
    for (const auto &entry : classes) {
        const vector<int> &services = entry.second;
        for (size_t index = 1; index < services.size(); ++index) {
            vector<literal> earlier(positions, formula->falseVar());
            vector<literal> later(positions, formula->falseVar());
            for (int agent = 0; agent < instance->A; ++agent) {
                for (int slot = 0; slot < instance->TS; ++slot) {
                    const int position = agent * instance->TS + slot;
                    earlier[position] = x[agent][services[index - 1]][slot];
                    later[position] = x[agent][services[index]][slot];
                }
            }
            addValuePrecedence(formula, earlier, later);
        }
    }
}

void HCORAPMultiObjectiveEncoding::addAgentSymmetryBreaking(
    SMTFormula *formula
) {
    map<vector<int>, vector<int> > classes;
    for (int agent = 0; agent < instance->A; ++agent) {
        vector<int> signature;
        signature.push_back(instance->HN[agent]);
        signature.push_back(instance->HE[agent]);
        for (int service = 0; service < instance->S; ++service)
            signature.push_back(instance->r[agent][service]);
        for (int service = 0; service < instance->S; ++service) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                signature.push_back(
                    x[agent][service][slot].v.id != formula->falseVar().id
                );
            }
        }
        classes[signature].push_back(agent);
    }

    for (const auto &entry : classes) {
        const vector<int> &agents = entry.second;
        for (size_t index = 1; index < agents.size(); ++index)
            addValuePrecedence(
                formula, y[agents[index - 1]], y[agents[index]]
            );
    }
}

int HCORAPMultiObjectiveEncoding::effectiveWorkloadCapacity(int agent) const {
    int candidateServices = 0;
    set<int> usableSlots;
    for (int service = 0; service < instance->S; ++service) {
        bool feasible = false;
        for (int slot = 0; slot < instance->TS; ++slot) {
            if (instance->r[agent][service] > 0 &&
                instance->TSA[agent][slot] &&
                instance->TSS[service][slot]) {
                feasible = true;
                usableSlots.insert(slot);
            }
        }
        if (feasible)
            ++candidateServices;
    }
    return min(
        instance->HN[agent] + instance->HE[agent],
        min(candidateServices, static_cast<int>(usableSlots.size()))
    );
}

void HCORAPMultiObjectiveEncoding::addPBAtLeast(
    SMTFormula *formula,
    const vector<int> &weights,
    const vector<literal> &values,
    int lowerBound
) {
    vector<literal> negated;
    negated.reserve(values.size());
    int totalWeight = 0;
    for (size_t index = 0; index < values.size(); ++index) {
        negated.push_back(!values[index]);
        totalWeight += weights[index];
    }
    formula->addPB(weights, negated, totalWeight - lowerBound);
}

SMTFormula *HCORAPMultiObjectiveEncoding::encode(int, int) {
    SMTFormula *formula = new SMTFormula();

    x = vector<vector<vector<literal> > >(
        instance->A,
        vector<vector<literal> >(instance->S, vector<literal>(instance->TS))
    );
    y = vector<vector<literal> >(
        instance->A, vector<literal>(instance->S)
    );
    performed = vector<literal>(instance->S);
    sequenceAgent = vector<vector<literal> >(
        instance->A, vector<literal>(instance->SEQ.size())
    );
    sequenceActive = vector<literal>(instance->SEQ.size());
    overtimeThreshold = vector<vector<literal> >(instance->A);

    // Only feasible assignment triples receive a decision variable.
    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                x[agent][service][slot] =
                    instance->r[agent][service] == 0 ||
                    !instance->TSA[agent][slot] ||
                    !instance->TSS[service][slot]
                    ? formula->falseVar()
                    : formula->newBoolVar("x", agent, service, slot);
            }
        }
    }

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            y[agent][service] = instance->r[agent][service] == 0
                ? formula->falseVar()
                : formula->newBoolVar("y", agent, service);
        }
    }
    for (int service = 0; service < instance->S; ++service)
        performed[service] = formula->newBoolVar("performed", service);
    for (int sequence = 0; sequence < static_cast<int>(instance->SEQ.size()); ++sequence) {
        sequenceActive[sequence] = formula->newBoolVar("sequence_active", sequence);
        for (int agent = 0; agent < instance->A; ++agent)
            sequenceAgent[agent][sequence] = formula->newBoolVar(
                "sequence_agent", agent, sequence
            );
    }

    serviceSlot.clear();
    userUsedSlot.clear();
    const bool breakSlotSymmetry =
        hcorapBreaksSlotSymmetry(symmetryBreaking) && hasSlotSymmetry();
    if (impliedConfig != HCORAP_IMPLIED_NONE ||
        breakSlotSymmetry)
        addServiceSlotVariables(formula);

    // y[a,s] <-> OR_t x[a,s,t].
    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            if (y[agent][service].v.id == formula->falseVar().id)
                continue;
            clause reverse = !y[agent][service];
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (x[agent][service][slot].v.id == formula->falseVar().id)
                    continue;
                formula->addClause(!x[agent][service][slot] | y[agent][service]);
                reverse |= x[agent][service][slot];
            }
            formula->addClause(reverse);
        }
    }

    // performed[s] <-> OR_{a,t} x[a,s,t], with at-most-one assignment.
    for (int service = 0; service < instance->S; ++service) {
        vector<literal> candidates;
        clause reverse = !performed[service];
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (x[agent][service][slot].v.id == formula->falseVar().id)
                    continue;
                candidates.push_back(x[agent][service][slot]);
                reverse |= x[agent][service][slot];
                formula->addClause(
                    !x[agent][service][slot] | performed[service]
                );
            }
        }
        formula->addClause(reverse);
        addAtMostOne(formula, candidates);
        if (fullCoverage)
            formula->addClause(performed[service]);
    }
    if (hcorapUsesPlusImprovements(impliedConfig))
        addProjectedServiceAssignments(formula);

    // At most one service per agent and time slot.
    for (int agent = 0; agent < instance->A; ++agent) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            vector<literal> values;
            for (int service = 0; service < instance->S; ++service) {
                if (x[agent][service][slot].v.id != formula->falseVar().id)
                    values.push_back(x[agent][service][slot]);
            }
            addAtMostOne(formula, values);
        }
    }

    // At most one simultaneous service for a user.
    for (const vector<int> &services : instance->SU) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            vector<literal> values;
            if (hcorapUsesPlusImprovements(impliedConfig)) {
                for (int service : services) {
                    if (serviceSlot[service][slot].v.id != formula->falseVar().id)
                        values.push_back(serviceSlot[service][slot]);
                }
            } else {
                for (int service : services) {
                    for (int agent = 0; agent < instance->A; ++agent) {
                        if (x[agent][service][slot].v.id != formula->falseVar().id)
                            values.push_back(x[agent][service][slot]);
                    }
                }
            }
            addAtMostOne(formula, values);
        }
    }

    if (hcorapUsesUserSlots(impliedConfig))
        addUserSlotConstraints(formula);
    if (hcorapUsesSlotCapacity(impliedConfig))
        addSlotCapacityConstraints(formula);

    if (breakSlotSymmetry)
        addSlotSymmetryBreaking(formula);
    if (hcorapBreaksServiceSymmetry(symmetryBreaking))
        addServiceSymmetryBreaking(formula);
    if (hcorapBreaksAgentSymmetry(symmetryBreaking))
        addAgentSymmetryBreaking(formula);

    // sequenceAgent[a,q] <-> OR_{s in q} y[a,s].
    for (int sequence = 0; sequence < static_cast<int>(instance->SEQ.size()); ++sequence) {
        for (int agent = 0; agent < instance->A; ++agent) {
            clause reverse = !sequenceAgent[agent][sequence];
            for (int service : instance->SEQ[sequence]) {
                if (y[agent][service].v.id == formula->falseVar().id)
                    continue;
                formula->addClause(
                    !y[agent][service] | sequenceAgent[agent][sequence]
                );
                reverse |= y[agent][service];
            }
            formula->addClause(reverse);
        }

        // sequenceActive[q] <-> OR_{s in q} performed[s].
        clause activeReverse = !sequenceActive[sequence];
        for (int service : instance->SEQ[sequence]) {
            formula->addClause(
                !performed[service] | sequenceActive[sequence]
            );
            activeReverse |= performed[service];
        }
        formula->addClause(activeReverse);
    }

    // Exact workload thresholds and maximum workload per agent.
    for (int agent = 0; agent < instance->A; ++agent) {
        vector<literal> workload;
        for (int service = 0; service < instance->S; ++service) {
            if (y[agent][service].v.id == formula->falseVar().id)
                continue;
            if (hcorapUsesPlusImprovements(impliedConfig)) {
                bool feasible = false;
                for (int slot = 0; slot < instance->TS; ++slot) {
                    if (x[agent][service][slot].v.id != formula->falseVar().id) {
                        feasible = true;
                        break;
                    }
                }
                if (!feasible)
                    continue;
            }
            workload.push_back(y[agent][service]);
        }
        if (workload.empty())
            continue;
        vector<literal> workloadThresholds;
        addHCORAPCardinalityNetwork(
            formula, workload, workloadThresholds, cardinalityEncoding
        );
        int maximum = hcorapUsesPlusImprovements(impliedConfig)
            ? effectiveWorkloadCapacity(agent)
            : instance->HN[agent] + instance->HE[agent];
        if (maximum < static_cast<int>(workloadThresholds.size()))
            formula->addClause(!workloadThresholds[maximum]);
        int last = min(maximum, static_cast<int>(workloadThresholds.size()));
        for (int threshold = instance->HN[agent]; threshold < last; ++threshold)
            overtimeThreshold[agent].push_back(workloadThresholds[threshold]);
    }

    addBounds(formula);
    addObjective(formula);
    return formula;
}

void HCORAPMultiObjectiveEncoding::addBounds(SMTFormula *formula) {
    vector<int> weights;
    vector<literal> values;

    if (bounds.minCoverage >= 0) {
        weights.assign(performed.size(), 1);
        addPBAtLeast(formula, weights, performed, bounds.minCoverage);
    }

    if (bounds.minSimilarity >= 0) {
        weights.clear();
        values.clear();
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int service = 0; service < instance->S; ++service) {
                if (instance->r[agent][service] > 0) {
                    weights.push_back(instance->r[agent][service]);
                    values.push_back(y[agent][service]);
                }
            }
        }
        if (!values.empty()) {
            addPBAtLeast(formula, weights, values, bounds.minSimilarity);
        } else if (bounds.minSimilarity > 0) {
            formula->addClause(formula->falseVar());
        }
    }

    if (bounds.maxContinuity >= 0) {
        weights.clear();
        values.clear();
        for (int sequence = 0; sequence < static_cast<int>(instance->SEQ.size()); ++sequence) {
            for (int agent = 0; agent < instance->A; ++agent) {
                weights.push_back(1);
                values.push_back(sequenceAgent[agent][sequence]);
            }
            weights.push_back(1);
            values.push_back(!sequenceActive[sequence]);
        }
        if (!values.empty()) {
            formula->addPB(
                weights,
                values,
                bounds.maxContinuity + static_cast<int>(instance->SEQ.size())
            );
        }
    }

    if (bounds.maxOvertime >= 0) {
        weights.clear();
        values.clear();
        for (const vector<literal> &agentThresholds : overtimeThreshold) {
            for (const literal &threshold : agentThresholds) {
                weights.push_back(1);
                values.push_back(threshold);
            }
        }
        if (!values.empty())
            formula->addPB(weights, values, bounds.maxOvertime);
    }
}

void HCORAPMultiObjectiveEncoding::addObjective(SMTFormula *formula) {
    int softCount = 0;
    if (objective == HCORAP_WEIGHTED || objective == HCORAP_SIMILARITY) {
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int service = 0; service < instance->S; ++service) {
                if (instance->r[agent][service] > 0) {
                    formula->addSoftClause(
                        y[agent][service], instance->r[agent][service]
                    );
                    ++softCount;
                }
            }
        }
    }

    if (objective == HCORAP_WEIGHTED || objective == HCORAP_CONTINUITY) {
        int weight = objective == HCORAP_WEIGHTED ? continuityWeight : 1;
        if (weight > 0) {
            for (int sequence = 0; sequence < static_cast<int>(instance->SEQ.size()); ++sequence) {
                for (int agent = 0; agent < instance->A; ++agent) {
                    formula->addSoftClause(
                        !sequenceAgent[agent][sequence], weight
                    );
                    ++softCount;
                }
                formula->addSoftClause(sequenceActive[sequence], weight);
                ++softCount;
            }
        }
    }

    if (objective == HCORAP_WEIGHTED || objective == HCORAP_OVERTIME) {
        int weight = objective == HCORAP_WEIGHTED
            ? overtimeWeight * abs(instance->P)
            : 1;
        if (weight > 0) {
            for (const vector<literal> &agentThresholds : overtimeThreshold) {
                for (const literal &threshold : agentThresholds) {
                    formula->addSoftClause(!threshold, weight);
                    ++softCount;
                }
            }
        }
    }

    if (objective == HCORAP_COVERAGE) {
        for (const literal &value : performed) {
            formula->addSoftClause(value, 1);
            ++softCount;
        }
    }

    // Keep a valid MaxSAT formula even for a constant objective.
    if (softCount == 0)
        formula->addSoftClause(formula->trueVar(), 1);
}

void HCORAPMultiObjectiveEncoding::setBooleanModel(
    const vector<bool> &values
) {
    model = values;
}

bool HCORAPMultiObjectiveEncoding::literalValue(const literal &value) const {
    if (value.v.id <= 0 || value.v.id >= static_cast<int>(model.size()))
        return false;
    bool variable = model[value.v.id];
    return value.sign ? variable : !variable;
}

HCORAPSolutionMetrics HCORAPMultiObjectiveEncoding::evaluateModel() const {
    HCORAPSolutionMetrics metrics;
    if (model.empty())
        return metrics;

    metrics.valid = true;
    metrics.workload.assign(instance->A, 0);
    vector<int> serviceCount(instance->S, 0);
    vector<vector<int> > serviceAtSlot(
        instance->S, vector<int>(instance->TS, 0)
    );
    vector<vector<bool> > assigned(
        instance->A, vector<bool>(instance->S, false)
    );
    vector<vector<int> > agentSlot(
        instance->A, vector<int>(instance->TS, 0)
    );
    vector<int> serviceUser(instance->S, -1);
    for (int user = 0; user < static_cast<int>(instance->SU.size()); ++user) {
        for (int service : instance->SU[user])
            serviceUser[service] = user;
    }
    vector<vector<int> > userSlot(
        instance->SU.size(), vector<int>(instance->TS, 0)
    );

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (x[agent][service][slot].v.id <= 0 ||
                    !literalValue(x[agent][service][slot]))
                    continue;
                metrics.assignments.push_back(make_tuple(agent, service, slot));
                assigned[agent][service] = true;
                ++serviceCount[service];
                ++serviceAtSlot[service][slot];
                ++agentSlot[agent][slot];
                ++metrics.workload[agent];
                if (serviceUser[service] >= 0)
                    ++userSlot[serviceUser[service]][slot];
                if (instance->r[agent][service] <= 0 ||
                    !instance->TSA[agent][slot] ||
                    !instance->TSS[service][slot])
                    metrics.valid = false;
            }
        }
    }

    for (int service = 0; service < instance->S; ++service) {
        bool isPerformed = serviceCount[service] > 0;
        if (isPerformed)
            ++metrics.coverage;
        if (serviceCount[service] > 1)
            metrics.valid = false;
        if (literalValue(performed[service]) != isPerformed)
            metrics.valid = false;
    }
    if (!serviceSlot.empty()) {
        for (int service = 0; service < instance->S; ++service) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                bool expected = serviceAtSlot[service][slot] > 0;
                if (literalValue(serviceSlot[service][slot]) != expected)
                    metrics.valid = false;
            }
        }
    }
    if (fullCoverage && metrics.coverage != instance->S)
        metrics.valid = false;

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            if (agentSlot[agent][slot] > 1)
                metrics.valid = false;
        }
        if (metrics.workload[agent] > instance->HN[agent] + instance->HE[agent])
            metrics.valid = false;
        metrics.overtime += max(0, metrics.workload[agent] - instance->HN[agent]);
        for (int service = 0; service < instance->S; ++service) {
            if (assigned[agent][service])
                metrics.similarity += instance->r[agent][service];
            if (literalValue(y[agent][service]) != assigned[agent][service])
                metrics.valid = false;
        }
        for (size_t index = 0; index < overtimeThreshold[agent].size(); ++index) {
            bool expected = metrics.workload[agent]
                > instance->HN[agent] + static_cast<int>(index);
            if (literalValue(overtimeThreshold[agent][index]) != expected)
                metrics.valid = false;
        }
    }
    for (const vector<int> &slots : userSlot) {
        for (int count : slots) {
            if (count > 1)
                metrics.valid = false;
        }
    }
    if (!userUsedSlot.empty()) {
        for (int user = 0; user < static_cast<int>(userUsedSlot.size()); ++user) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                bool expected = userSlot[user][slot] > 0;
                if (literalValue(userUsedSlot[user][slot]) != expected)
                    metrics.valid = false;
            }
        }
    }

    for (int sequence = 0; sequence < static_cast<int>(instance->SEQ.size()); ++sequence) {
        int distinctAgents = 0;
        bool active = false;
        for (int service : instance->SEQ[sequence]) {
            if (serviceCount[service] > 0)
                active = true;
        }
        for (int agent = 0; agent < instance->A; ++agent) {
            bool participates = false;
            for (int service : instance->SEQ[sequence]) {
                if (assigned[agent][service]) {
                    participates = true;
                    break;
                }
            }
            if (participates)
                ++distinctAgents;
            if (literalValue(sequenceAgent[agent][sequence]) != participates)
                metrics.valid = false;
        }
        if (literalValue(sequenceActive[sequence]) != active)
            metrics.valid = false;
        if (active)
            metrics.continuity += max(0, distinctAgents - 1);
    }
    metrics.overtimeCost = abs(instance->P) * metrics.overtime;
    return metrics;
}

int HCORAPMultiObjectiveEncoding::objectiveValue(
    const HCORAPSolutionMetrics &metrics
) const {
    switch (objective) {
        case HCORAP_COVERAGE:
            return metrics.coverage;
        case HCORAP_SIMILARITY:
            return metrics.similarity;
        case HCORAP_CONTINUITY:
            return metrics.continuity;
        case HCORAP_OVERTIME:
            return metrics.overtime;
        case HCORAP_WEIGHTED:
        default:
            return metrics.similarity
                - continuityWeight * metrics.continuity
                - overtimeWeight * abs(instance->P) * metrics.overtime;
    }
}
