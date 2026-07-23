#include "CommercialTypes.h"
#include "HCORAPMIPModel.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <stdexcept>

using namespace std;

namespace {

class ReferenceBackend : public HCORAPCommercialBackend {
    HCORAPBackendConfig config;

public:
    explicit ReferenceBackend(const HCORAPBackendConfig &config)
        : config(config) {}

    string name() const { return "reference-enumerator"; }
    string formulation() const { return "direct-schedule-enumeration"; }
    string version() const { return "1"; }

    HCORAPStageResult solve(const HCORAPStageRequest &request) {
        HCORAPStageResult result;
        auto buildStarted = chrono::steady_clock::now();
        HCORAPMIPModel mip = buildHCORAPMIPModel(request);
        result.variables = static_cast<int>(mip.variables.size());
        result.constraints = static_cast<int>(mip.constraints.size());

        const HCORAP *instance = request.instance;
        vector<vector<tuple<int, int, int> > > choices(instance->S);
        for (int service = 0; service < instance->S; ++service) {
            if (!request.fullCoverage)
                choices[service].push_back(make_tuple(-1, service, -1));
            for (int agent = 0; agent < instance->A; ++agent) {
                if (instance->r[agent][service] <= 0)
                    continue;
                for (int slot = 0; slot < instance->TS; ++slot) {
                    if (instance->TSA[agent][slot] &&
                        instance->TSS[service][slot]) {
                        choices[service].push_back(
                            make_tuple(agent, service, slot)
                        );
                    }
                }
            }
            if (request.fullCoverage && choices[service].empty()) {
                result.status = COMMERCIAL_INFEASIBLE;
                result.buildSeconds = chrono::duration<double>(
                    chrono::steady_clock::now() - buildStarted
                ).count();
                return result;
            }
        }

        vector<int> order(instance->S);
        for (int service = 0; service < instance->S; ++service)
            order[service] = service;
        sort(order.begin(), order.end(), [&](int left, int right) {
            if (choices[left].size() != choices[right].size())
                return choices[left].size() < choices[right].size();
            return left < right;
        });

        result.buildSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - buildStarted
        ).count();
        const double solveBudget =
            request.timeoutSeconds - result.buildSeconds;
        if (solveBudget <= 0) {
            result.status = COMMERCIAL_TIMEOUT;
            result.message =
                "cumulative timeout exhausted during model build";
            return result;
        }
        auto solveStarted = chrono::steady_clock::now();
        vector<vector<bool> > agentSlot(
            instance->A, vector<bool>(instance->TS, false)
        );
        vector<vector<bool> > userSlot(
            instance->SU.size(), vector<bool>(instance->TS, false)
        );
        vector<int> serviceUser(instance->S, -1);
        for (size_t user = 0; user < instance->SU.size(); ++user) {
            for (int service : instance->SU[user])
                serviceUser[service] = static_cast<int>(user);
        }
        vector<int> workload(instance->A, 0);
        vector<tuple<int, int, int> > current;
        vector<tuple<int, int, int> > best;
        bool incumbent = false;
        bool stopped = false;
        int bestValue = request.objective == COMMERCIAL_CONTINUITY ||
            request.objective == COMMERCIAL_OVERTIME
            ? numeric_limits<int>::max()
            : numeric_limits<int>::min();

        function<void(size_t)> search = [&](size_t depth) {
            if (stopped)
                return;
            if (result.explored >= config.enumerationLimit) {
                stopped = true;
                result.message = "reference enumeration limit reached";
                return;
            }
            double elapsed = chrono::duration<double>(
                chrono::steady_clock::now() - solveStarted
            ).count();
            if (elapsed >= solveBudget) {
                stopped = true;
                result.message = "reference enumeration timeout";
                return;
            }
            if (depth == order.size()) {
                ++result.explored;
                HCORAPCommercialMetrics metrics = verifyHCORAPAssignments(
                    instance, current, request.fullCoverage
                );
                if (!metrics.valid ||
                    !hcorapCommercialBoundsSatisfied(request.bounds, metrics))
                    return;
                int value = hcorapCommercialObjectiveValue(
                    request.objective,
                    metrics,
                    instance,
                    request.continuityWeight,
                    request.overtimeWeight
                );
                long long linearValue = 0;
                const vector<string> linearViolations =
                    validateHCORAPMIPSchedule(
                        mip, request, current, &linearValue
                    );
                if (!linearViolations.empty())
                    throw runtime_error(
                        "MIP-E invariant failed: " +
                        linearViolations.front()
                    );
                if (linearValue != value)
                    throw runtime_error(
                        "MIP-E objective disagrees with verifier"
                    );
                bool better = !incumbent ||
                    ((request.objective == COMMERCIAL_CONTINUITY ||
                      request.objective == COMMERCIAL_OVERTIME)
                        ? value < bestValue : value > bestValue);
                if (better) {
                    incumbent = true;
                    bestValue = value;
                    best = current;
                }
                return;
            }

            const int service = order[depth];
            for (const tuple<int, int, int> &choice : choices[service]) {
                const int agent = get<0>(choice);
                const int slot = get<2>(choice);
                if (agent < 0) {
                    search(depth + 1);
                    continue;
                }
                const int user = serviceUser[service];
                const int capacity =
                    instance->HN[agent] + instance->HE[agent];
                if (agentSlot[agent][slot] ||
                    (user >= 0 && userSlot[user][slot]) ||
                    workload[agent] >= capacity)
                    continue;

                agentSlot[agent][slot] = true;
                if (user >= 0)
                    userSlot[user][slot] = true;
                ++workload[agent];
                current.push_back(choice);
                search(depth + 1);
                current.pop_back();
                --workload[agent];
                if (user >= 0)
                    userSlot[user][slot] = false;
                agentSlot[agent][slot] = false;
                if (stopped)
                    break;
            }
        };

        search(0);
        result.solveSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - solveStarted
        ).count();
        if (stopped) {
            result.status = incumbent
                ? COMMERCIAL_TIMEOUT_FEASIBLE : COMMERCIAL_TIMEOUT;
        } else if (!incumbent) {
            result.status = COMMERCIAL_INFEASIBLE;
        } else {
            result.status = COMMERCIAL_OPTIMUM;
            result.bestBound = bestValue;
            result.relativeGap = 0.0;
            result.hasBestBound = true;
            result.hasRelativeGap = true;
        }
        if (incumbent)
            result.assignments = best;
        return result;
    }
};

}

unique_ptr<HCORAPCommercialBackend> createReferenceBackend(
    const HCORAPBackendConfig &config
) {
    return unique_ptr<HCORAPCommercialBackend>(
        new ReferenceBackend(config)
    );
}
