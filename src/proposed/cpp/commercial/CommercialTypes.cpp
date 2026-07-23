#include "CommercialTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <limits>
#include <set>
#include <sstream>

using namespace std;

HCORAPCommercialBounds::HCORAPCommercialBounds()
    : minCoverage(-1), minSimilarity(-1), maxContinuity(-1),
      maxOvertime(-1) {}

HCORAPCommercialMetrics::HCORAPCommercialMetrics()
    : valid(false), coverage(0), similarity(0), continuity(0), overtime(0),
      overtimeCost(0) {}

HCORAPBackendConfig::HCORAPBackendConfig()
    : threads(1), seed(0), mipGap(0.0), absoluteMipGap(0.0),
      enumerationLimit(5000000) {}

HCORAPStageRequest::HCORAPStageRequest()
    : instance(NULL), objective(COMMERCIAL_WEIGHTED), fullCoverage(true),
      continuityWeight(1), overtimeWeight(1), timeoutSeconds(3600.0),
      stageIndex(0) {}

HCORAPStageResult::HCORAPStageResult()
    : status(COMMERCIAL_ERROR), buildSeconds(0.0), solveSeconds(0.0),
      bestBound(0.0), relativeGap(0.0), hasBestBound(false),
      hasRelativeGap(false), variables(0), constraints(0), explored(0) {}

const char *hcorapCommercialObjectiveName(
    HCORAPCommercialObjective objective
) {
    switch (objective) {
        case COMMERCIAL_COVERAGE: return "coverage";
        case COMMERCIAL_SIMILARITY: return "similarity";
        case COMMERCIAL_CONTINUITY: return "continuity";
        case COMMERCIAL_OVERTIME: return "overtime";
        case COMMERCIAL_WEIGHTED:
        default: return "weighted_score";
    }
}

const char *hcorapCommercialObjectiveSense(
    HCORAPCommercialObjective objective
) {
    return objective == COMMERCIAL_CONTINUITY ||
        objective == COMMERCIAL_OVERTIME ? "min" : "max";
}

const char *hcorapCommercialStatusName(HCORAPCommercialStatus status) {
    switch (status) {
        case COMMERCIAL_OPTIMUM: return "OPTIMUM";
        case COMMERCIAL_INFEASIBLE: return "INFEASIBLE";
        case COMMERCIAL_TIMEOUT_FEASIBLE: return "TIMEOUT_FEASIBLE";
        case COMMERCIAL_TIMEOUT: return "TIMEOUT";
        case COMMERCIAL_ERROR:
        default: return "ERROR";
    }
}

int hcorapCommercialObjectiveValue(
    HCORAPCommercialObjective objective,
    const HCORAPCommercialMetrics &metrics,
    const HCORAP *instance,
    int continuityWeight,
    int overtimeWeight
) {
    switch (objective) {
        case COMMERCIAL_COVERAGE:
            return metrics.coverage;
        case COMMERCIAL_SIMILARITY:
            return metrics.similarity;
        case COMMERCIAL_CONTINUITY:
            return metrics.continuity;
        case COMMERCIAL_OVERTIME:
            return metrics.overtime;
        case COMMERCIAL_WEIGHTED:
        default:
            return metrics.similarity
                - continuityWeight * metrics.continuity
                - overtimeWeight * abs(instance->P) * metrics.overtime;
    }
}

bool hcorapCommercialBoundsSatisfied(
    const HCORAPCommercialBounds &bounds,
    const HCORAPCommercialMetrics &metrics
) {
    return
        (bounds.minCoverage < 0 || metrics.coverage >= bounds.minCoverage) &&
        (bounds.minSimilarity < 0 ||
         metrics.similarity >= bounds.minSimilarity) &&
        (bounds.maxContinuity < 0 ||
         metrics.continuity <= bounds.maxContinuity) &&
        (bounds.maxOvertime < 0 || metrics.overtime <= bounds.maxOvertime);
}

vector<string> validateHCORAPInstance(const HCORAP *instance) {
    vector<string> violations;
    if (instance == NULL) {
        violations.push_back("instance is null");
        return violations;
    }
    if (instance->U < 0 || instance->S < 0 ||
        instance->A < 0 || instance->TS < 0)
        violations.push_back("instance dimensions must be non-negative");
    if (instance->P == numeric_limits<int>::min())
        violations.push_back("overtime penalty magnitude is out of range");
    if (static_cast<int>(instance->TSA.size()) != instance->A)
        violations.push_back("TSA row count does not match A");
    if (static_cast<int>(instance->TSS.size()) != instance->S)
        violations.push_back("TSS row count does not match S");
    if (static_cast<int>(instance->r.size()) != instance->A)
        violations.push_back("reward row count does not match A");
    if (static_cast<int>(instance->HN.size()) != instance->A ||
        static_cast<int>(instance->HE.size()) != instance->A)
        violations.push_back("HN/HE length does not match A");
    if (static_cast<int>(instance->SU.size()) != instance->U)
        violations.push_back("SU row count does not match U");

    for (int agent = 0; agent < instance->A; ++agent) {
        if (agent < static_cast<int>(instance->TSA.size()) &&
            static_cast<int>(instance->TSA[agent].size()) != instance->TS)
            violations.push_back("TSA column count does not match TS");
        if (agent < static_cast<int>(instance->r.size()) &&
            static_cast<int>(instance->r[agent].size()) != instance->S)
            violations.push_back("reward column count does not match S");
        if (agent < static_cast<int>(instance->r.size())) {
            for (int reward : instance->r[agent]) {
                if (reward < 0)
                    violations.push_back(
                        "reward values must be non-negative"
                    );
            }
        }
        if (agent < static_cast<int>(instance->HN.size()) &&
            agent < static_cast<int>(instance->HE.size()) &&
            (instance->HN[agent] < 0 || instance->HE[agent] < 0))
            violations.push_back("HN/HE values must be non-negative");
        if (agent < static_cast<int>(instance->HN.size()) &&
            agent < static_cast<int>(instance->HE.size()) &&
            static_cast<long long>(instance->HN[agent]) +
                instance->HE[agent] > numeric_limits<int>::max())
            violations.push_back("HN+HE exceeds the integer range");
    }
    for (int service = 0; service < instance->S; ++service) {
        if (service < static_cast<int>(instance->TSS.size()) &&
            static_cast<int>(instance->TSS[service].size()) != instance->TS)
            violations.push_back("TSS column count does not match TS");
    }

    vector<int> serviceUser(instance->S, -1);
    for (size_t user = 0; user < instance->SU.size(); ++user) {
        for (int service : instance->SU[user]) {
            if (service < 0 || service >= instance->S) {
                violations.push_back("SU contains an invalid service index");
            } else if (serviceUser[service] != -1) {
                violations.push_back("SU is not a partition: duplicate service");
            } else {
                serviceUser[service] = static_cast<int>(user);
            }
        }
    }
    for (int service = 0; service < instance->S; ++service) {
        if (serviceUser[service] < 0)
            violations.push_back("SU is not a partition: missing service");
    }
    for (const vector<int> &sequence : instance->SEQ) {
        set<int> seen;
        for (int service : sequence) {
            if (service < 0 || service >= instance->S)
                violations.push_back("SEQ contains an invalid service index");
            else if (!seen.insert(service).second)
                violations.push_back("SEQ contains a duplicate service");
        }
    }
    return violations;
}

HCORAPCommercialMetrics verifyHCORAPAssignments(
    const HCORAP *instance,
    const vector<tuple<int, int, int> > &assignments,
    bool requireFullCoverage
) {
    HCORAPCommercialMetrics metrics;
    metrics.valid = true;
    metrics.assignments = assignments;
    metrics.workload.assign(instance->A, 0);

    vector<int> serviceCount(instance->S, 0);
    vector<vector<int> > agentSlot(
        instance->A, vector<int>(instance->TS, 0)
    );
    vector<int> serviceUser(instance->S, -1);
    for (size_t user = 0; user < instance->SU.size(); ++user) {
        for (int service : instance->SU[user]) {
            if (service >= 0 && service < instance->S)
                serviceUser[service] = static_cast<int>(user);
        }
    }
    vector<vector<int> > userSlot(
        instance->SU.size(), vector<int>(instance->TS, 0)
    );
    vector<vector<bool> > assigned(
        instance->A, vector<bool>(instance->S, false)
    );
    set<tuple<int, int, int> > uniqueAssignments;

    for (const tuple<int, int, int> &assignment : assignments) {
        const int agent = get<0>(assignment);
        const int service = get<1>(assignment);
        const int slot = get<2>(assignment);
        if (!uniqueAssignments.insert(assignment).second)
            metrics.violations.push_back("duplicate assignment triple");
        if (agent < 0 || agent >= instance->A ||
            service < 0 || service >= instance->S ||
            slot < 0 || slot >= instance->TS) {
            metrics.violations.push_back("assignment index out of range");
            continue;
        }
        if (instance->r[agent][service] <= 0)
            metrics.violations.push_back("unqualified agent-service assignment");
        if (!instance->TSA[agent][slot])
            metrics.violations.push_back("agent assigned outside availability");
        if (!instance->TSS[service][slot])
            metrics.violations.push_back("service assigned outside time window");

        ++serviceCount[service];
        ++agentSlot[agent][slot];
        ++metrics.workload[agent];
        assigned[agent][service] = true;
        if (serviceUser[service] >= 0)
            ++userSlot[serviceUser[service]][slot];
        metrics.similarity += instance->r[agent][service];
    }

    for (int service = 0; service < instance->S; ++service) {
        if (serviceCount[service] > 0)
            ++metrics.coverage;
        if (serviceCount[service] > 1)
            metrics.violations.push_back("service assigned more than once");
        if (requireFullCoverage && serviceCount[service] != 1)
            metrics.violations.push_back("full-coverage service is unassigned");
    }
    for (int agent = 0; agent < instance->A; ++agent) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            if (agentSlot[agent][slot] > 1)
                metrics.violations.push_back(
                    "agent performs simultaneous services"
                );
        }
        const int capacity = instance->HN[agent] + instance->HE[agent];
        if (metrics.workload[agent] > capacity)
            metrics.violations.push_back("agent workload exceeds capacity");
        metrics.overtime += max(
            0, metrics.workload[agent] - instance->HN[agent]
        );
    }
    for (const vector<int> &slots : userSlot) {
        for (int count : slots) {
            if (count > 1)
                metrics.violations.push_back(
                    "user receives simultaneous services"
                );
        }
    }

    for (const vector<int> &sequence : instance->SEQ) {
        set<int> agents;
        bool active = false;
        for (int service : sequence) {
            if (service < 0 || service >= instance->S)
                continue;
            if (serviceCount[service] > 0)
                active = true;
            for (int agent = 0; agent < instance->A; ++agent) {
                if (assigned[agent][service])
                    agents.insert(agent);
            }
        }
        if (active)
            metrics.continuity += max(0, static_cast<int>(agents.size()) - 1);
    }
    metrics.overtimeCost = abs(instance->P) * metrics.overtime;
    metrics.valid = metrics.violations.empty();
    return metrics;
}
