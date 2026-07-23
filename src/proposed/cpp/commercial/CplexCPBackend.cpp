#include "CommercialTypes.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef HCORAP_WITH_CPLEX
#include <ilcp/cp.h>
ILOSTLBEGIN
#endif

using namespace std;

#ifdef HCORAP_WITH_CPLEX

namespace {

static string cpName(
    const string &prefix,
    int first,
    int second = -1,
    int third = -1
) {
    ostringstream name;
    name << prefix << '_' << first;
    if (second >= 0)
        name << '_' << second;
    if (third >= 0)
        name << '_' << third;
    return name.str();
}

static bool cpCandidate(
    const HCORAP *instance,
    int agent,
    int service,
    int slot
) {
    return instance->r[agent][service] > 0 &&
        instance->TSA[agent][slot] &&
        instance->TSS[service][slot];
}

struct CPExpressions {
    IloIntExpr coverage;
    IloIntExpr similarity;
    IloIntExpr continuity;
    IloIntExpr overtime;

    explicit CPExpressions(const IloEnv environment)
        : coverage(environment), similarity(environment),
          continuity(environment), overtime(environment) {}
};

static int addCPBoundsAndObjective(
    const IloEnv environment,
    IloModel &model,
    const HCORAPStageRequest &request,
    CPExpressions &expressions
) {
    int constraints = 0;
    if (request.bounds.minCoverage >= 0) {
        model.add(expressions.coverage >= request.bounds.minCoverage);
        ++constraints;
    }
    if (request.bounds.minSimilarity >= 0) {
        model.add(expressions.similarity >= request.bounds.minSimilarity);
        ++constraints;
    }
    if (request.bounds.maxContinuity >= 0) {
        model.add(expressions.continuity <= request.bounds.maxContinuity);
        ++constraints;
    }
    if (request.bounds.maxOvertime >= 0) {
        model.add(expressions.overtime <= request.bounds.maxOvertime);
        ++constraints;
    }

    IloExpr objective(environment);
    bool maximize = true;
    switch (request.objective) {
        case COMMERCIAL_COVERAGE:
            objective += expressions.coverage;
            break;
        case COMMERCIAL_SIMILARITY:
            objective += expressions.similarity;
            break;
        case COMMERCIAL_CONTINUITY:
            objective += expressions.continuity;
            maximize = false;
            break;
        case COMMERCIAL_OVERTIME:
            objective += expressions.overtime;
            maximize = false;
            break;
        case COMMERCIAL_WEIGHTED:
        default:
            objective += expressions.similarity;
            objective -= request.continuityWeight * expressions.continuity;
            objective -= (
                static_cast<IloNum>(request.overtimeWeight) *
                abs(request.instance->P)
            ) * expressions.overtime;
            break;
    }
    if (maximize)
        model.add(IloMaximize(environment, objective));
    else
        model.add(IloMinimize(environment, objective));
    objective.end();
    return constraints;
}

static void addExactOvertime(
    const IloEnv environment,
    IloModel &model,
    const HCORAP *instance,
    int agent,
    const IloIntExpr &workload,
    CPExpressions &expressions,
    int &variables,
    int &constraints
) {
    const int capacity = instance->HN[agent] + instance->HE[agent];
    model.add(workload <= capacity);
    ++constraints;
    for (int unit = 1; unit <= instance->HE[agent]; ++unit) {
        const int threshold = instance->HN[agent] + unit;
        IloBoolVar overtime(
            environment, cpName("overtime", agent, unit).c_str()
        );
        model.add(workload >= threshold * overtime);
        model.add(
            workload <= threshold - 1 +
                (capacity - threshold + 1) * overtime
        );
        expressions.overtime += overtime;
        ++variables;
        constraints += 2;
    }
}

static void configureCP(
    IloCP &cp,
    const HCORAPBackendConfig &config,
    double timeoutSeconds,
    ofstream &log,
    const IloEnv environment
) {
    cp.setParameter(IloCP::Workers, config.threads);
    cp.setParameter(IloCP::RandomSeed, config.seed);
    cp.setParameter(IloCP::TimeLimit, timeoutSeconds);
    cp.setParameter(IloCP::TimeMode, IloCP::ElapsedTime);
    cp.setParameter(IloCP::OptimalityTolerance, 0.0);
    cp.setParameter(IloCP::RelativeOptimalityTolerance, 0.0);
    if (config.solverLog.empty()) {
        cp.setParameter(IloCP::LogVerbosity, IloCP::Quiet);
        cp.setOut(environment.getNullStream());
    } else {
        log.open(config.solverLog.c_str(), ios::out | ios::app);
        if (!log)
            throw runtime_error(
                "cannot open CP Optimizer log: " + config.solverLog
            );
        cp.setOut(log);
        cp.setParameter(IloCP::LogVerbosity, IloCP::Normal);
    }
}

static void finishCPResult(
    IloCP &cp,
    bool solved,
    HCORAPStageResult &result
) {
    const IloAlgorithm::Status status = cp.getStatus();
    if (status == IloAlgorithm::Optimal) {
        result.status = COMMERCIAL_OPTIMUM;
    } else if (status == IloAlgorithm::Infeasible) {
        result.status = COMMERCIAL_INFEASIBLE;
    } else if (solved || status == IloAlgorithm::Feasible) {
        result.status = COMMERCIAL_TIMEOUT_FEASIBLE;
    } else {
        result.status = COMMERCIAL_TIMEOUT;
    }
    if (status != IloAlgorithm::Infeasible) {
        result.bestBound = cp.getObjBound();
        result.hasBestBound = true;
    }
    if (solved || status == IloAlgorithm::Feasible ||
        status == IloAlgorithm::Optimal) {
        result.relativeGap = cp.getObjGap();
        result.hasRelativeGap = true;
    }
    result.explored = static_cast<long long>(
        cp.getInfo(IloCP::NumberOfBranches)
    );
}

class CplexCPBackend : public HCORAPCommercialBackend {
    HCORAPBackendConfig config;
    IloEnv environment;
    string productVersion;

    HCORAPStageResult solveCPT(const HCORAPStageRequest &request) {
        HCORAPStageResult result;
        const HCORAP *instance = request.instance;
        auto buildStarted = chrono::steady_clock::now();
        IloModel model(environment);
        CPExpressions expressions(environment);

        const int maxReward = [&]() {
            int value = 0;
            for (const vector<int> &row : instance->r) {
                for (int reward : row)
                    value = max(value, reward);
            }
            return value;
        }();

        IloIntVarArray agentVars(environment);
        IloIntVarArray slotVars(environment);
        IloIntVarArray coveredVars(environment);
        IloIntVarArray rewardVars(environment);
        IloIntVarArray pairKeys(environment);
        IloIntVarArray userSlotKeys(environment);
        result.variables = 6 * instance->S;

        for (int service = 0; service < instance->S; ++service) {
            const int agentLower = request.fullCoverage ? 0 : -1;
            const int slotLower = request.fullCoverage ? 0 : -1;
            agentVars.add(IloIntVar(
                environment, agentLower, max(0, instance->A - 1),
                cpName("agent", service).c_str()
            ));
            slotVars.add(IloIntVar(
                environment, slotLower, max(0, instance->TS - 1),
                cpName("slot", service).c_str()
            ));
            coveredVars.add(IloIntVar(
                environment, request.fullCoverage ? 1 : 0, 1,
                cpName("covered", service).c_str()
            ));
            rewardVars.add(IloIntVar(
                environment, 0, maxReward,
                cpName("reward", service).c_str()
            ));
            pairKeys.add(IloIntVar(
                environment,
                0,
                instance->A * instance->TS + instance->S - 1,
                cpName("pair_key", service).c_str()
            ));
            userSlotKeys.add(IloIntVar(
                environment,
                0,
                instance->TS + instance->S - 1,
                cpName("user_slot_key", service).c_str()
            ));

            IloIntTupleSet tuples(environment, 6);
            IloIntArray tuple(environment, 6);
            int candidates = 0;
            for (int agent = 0; agent < instance->A; ++agent) {
                for (int slot = 0; slot < instance->TS; ++slot) {
                    if (!cpCandidate(instance, agent, service, slot))
                        continue;
                    tuple[0] = agent;
                    tuple[1] = slot;
                    tuple[2] = 1;
                    tuple[3] = instance->r[agent][service];
                    tuple[4] = agent * instance->TS + slot;
                    tuple[5] = slot;
                    tuples.add(tuple);
                    ++candidates;
                }
            }
            if (!request.fullCoverage) {
                tuple[0] = -1;
                tuple[1] = -1;
                tuple[2] = 0;
                tuple[3] = 0;
                tuple[4] = instance->A * instance->TS + service;
                tuple[5] = instance->TS + service;
                tuples.add(tuple);
            } else if (candidates == 0) {
                result.status = COMMERCIAL_INFEASIBLE;
                result.buildSeconds = chrono::duration<double>(
                    chrono::steady_clock::now() - buildStarted
                ).count();
                model.end();
                return result;
            }

            IloIntVarArray row(environment);
            row.add(agentVars[service]);
            row.add(slotVars[service]);
            row.add(coveredVars[service]);
            row.add(rewardVars[service]);
            row.add(pairKeys[service]);
            row.add(userSlotKeys[service]);
            model.add(IloAllowedAssignments(environment, row, tuples));
            ++result.constraints;
            expressions.coverage += coveredVars[service];
            expressions.similarity += rewardVars[service];
        }

        if (pairKeys.getSize() > 1) {
            model.add(IloAllDiff(environment, pairKeys));
            ++result.constraints;
        }
        for (const vector<int> &services : instance->SU) {
            IloIntVarArray keys(environment);
            for (int service : services)
                keys.add(userSlotKeys[service]);
            if (keys.getSize() > 1) {
                model.add(IloAllDiff(environment, keys));
                ++result.constraints;
            }
        }

        for (int agent = 0; agent < instance->A; ++agent) {
            IloIntExpr workload = IloCount(agentVars, agent);
            addExactOvertime(
                environment, model, instance, agent, workload,
                expressions, result.variables, result.constraints
            );
            workload.end();
        }

        for (size_t sequence = 0; sequence < instance->SEQ.size(); ++sequence) {
            IloIntVarArray sequenceCovered(environment);
            IloIntVarArray sequenceAgents(environment);
            for (int service : instance->SEQ[sequence]) {
                sequenceCovered.add(coveredVars[service]);
                sequenceAgents.add(agentVars[service]);
            }
            if (sequenceCovered.getSize() == 0)
                continue;

            IloBoolVar active(
                environment,
                cpName(
                    "sequence_active", static_cast<int>(sequence)
                ).c_str()
            );
            IloIntExpr activeCount(environment);
            for (IloInt index = 0;
                 index < sequenceCovered.getSize(); ++index)
                activeCount += sequenceCovered[index];
            model.add(activeCount >= active);
            model.add(
                activeCount <= sequenceCovered.getSize() * active
            );
            expressions.continuity -= active;
            ++result.variables;
            result.constraints += 2;

            for (int agent = 0; agent < instance->A; ++agent) {
                IloIntExpr count = IloCount(sequenceAgents, agent);
                IloBoolVar used(
                    environment,
                    cpName(
                        "sequence_agent",
                        agent,
                        static_cast<int>(sequence)
                    ).c_str()
                );
                model.add(count >= used);
                model.add(count <= sequenceAgents.getSize() * used);
                expressions.continuity += used;
                ++result.variables;
                result.constraints += 2;
                count.end();
            }
            activeCount.end();
        }

        result.constraints += addCPBoundsAndObjective(
            environment, model, request, expressions
        );
        result.buildSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - buildStarted
        ).count();
        const double solveBudget =
            request.timeoutSeconds - result.buildSeconds;
        if (solveBudget <= 0) {
            result.status = COMMERCIAL_TIMEOUT;
            result.message =
                "cumulative timeout exhausted during model build";
            model.end();
            return result;
        }

        IloCP cp(model);
        ofstream log;
        configureCP(
            cp, config, solveBudget, log, environment
        );
        auto solveStarted = chrono::steady_clock::now();
        const bool solved = cp.solve();
        result.solveSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - solveStarted
        ).count();
        finishCPResult(cp, solved, result);
        if (solved) {
            for (int service = 0; service < instance->S; ++service) {
                if (cp.getValue(coveredVars[service]) < 1)
                    continue;
                result.assignments.push_back(make_tuple(
                    static_cast<int>(cp.getValue(agentVars[service])),
                    service,
                    static_cast<int>(cp.getValue(slotVars[service]))
                ));
            }
        }
        cp.end();
        model.end();
        return result;
    }

    HCORAPStageResult solveCPI(const HCORAPStageRequest &request) {
        HCORAPStageResult result;
        const HCORAP *instance = request.instance;
        auto buildStarted = chrono::steady_clock::now();
        IloModel model(environment);
        CPExpressions expressions(environment);

        IloIntervalVarArray masters(environment);
        vector<IloIntervalVarArray> workerTasks;
        for (int agent = 0; agent < instance->A; ++agent)
            workerTasks.push_back(IloIntervalVarArray(environment));
        vector<vector<IloIntervalVar> > alternatives(
            instance->A, vector<IloIntervalVar>(instance->S)
        );
        vector<vector<bool> > exists(
            instance->A, vector<bool>(instance->S, false)
        );

        for (int service = 0; service < instance->S; ++service) {
            IloIntervalVar master(
                environment, 1, cpName("service", service).c_str()
            );
            master.setStartMin(0);
            master.setStartMax(max(0, instance->TS - 1));
            if (!request.fullCoverage)
                master.setOptional();
            masters.add(master);
            ++result.variables;

            IloIntervalVarArray serviceAlternatives(environment);
            for (int agent = 0; agent < instance->A; ++agent) {
                bool hasCandidate = false;
                IloNumToNumStepFunction calendar(
                    environment, 0, instance->TS, 0.0
                );
                for (int slot = 0; slot < instance->TS; ++slot) {
                    if (!cpCandidate(instance, agent, service, slot))
                        continue;
                    hasCandidate = true;
                    calendar.setValue(slot, slot + 1, 1.0);
                }
                if (!hasCandidate)
                    continue;

                IloIntervalVar alternative(
                    environment,
                    1,
                    cpName("agent_service", agent, service).c_str()
                );
                alternative.setOptional();
                alternative.setStartMin(0);
                alternative.setStartMax(max(0, instance->TS - 1));
                model.add(IloForbidStart(
                    environment, alternative, calendar
                ));
                serviceAlternatives.add(alternative);
                workerTasks[agent].add(alternative);
                alternatives[agent][service] = alternative;
                exists[agent][service] = true;
                expressions.similarity +=
                    instance->r[agent][service] *
                    IloPresenceOf(environment, alternative);
                ++result.variables;
                ++result.constraints;
            }

            if (serviceAlternatives.getSize() == 0) {
                if (request.fullCoverage) {
                    result.status = COMMERCIAL_INFEASIBLE;
                    result.buildSeconds = chrono::duration<double>(
                        chrono::steady_clock::now() - buildStarted
                    ).count();
                    model.end();
                    return result;
                }
                model.add(
                    IloPresenceOf(environment, master) == 0
                );
            } else {
                model.add(IloAlternative(
                    environment, master, serviceAlternatives
                ));
            }
            ++result.constraints;
            expressions.coverage += IloPresenceOf(environment, master);
        }

        for (int agent = 0; agent < instance->A; ++agent) {
            if (workerTasks[agent].getSize() > 1) {
                model.add(IloNoOverlap(
                    environment, workerTasks[agent]
                ));
                ++result.constraints;
            }
            IloIntExpr workload(environment);
            for (IloInt index = 0;
                 index < workerTasks[agent].getSize(); ++index)
                workload += IloPresenceOf(
                    environment, workerTasks[agent][index]
                );
            addExactOvertime(
                environment, model, instance, agent, workload,
                expressions, result.variables, result.constraints
            );
            workload.end();
        }

        for (const vector<int> &services : instance->SU) {
            IloIntervalVarArray userTasks(environment);
            for (int service : services)
                userTasks.add(masters[service]);
            if (userTasks.getSize() > 1) {
                model.add(IloNoOverlap(environment, userTasks));
                ++result.constraints;
            }
        }

        for (size_t sequence = 0; sequence < instance->SEQ.size(); ++sequence) {
            if (instance->SEQ[sequence].empty())
                continue;
            IloBoolVar active(
                environment,
                cpName(
                    "sequence_active", static_cast<int>(sequence)
                ).c_str()
            );
            IloIntExpr activeCount(environment);
            for (int service : instance->SEQ[sequence])
                activeCount += IloPresenceOf(environment, masters[service]);
            model.add(activeCount >= active);
            model.add(
                activeCount <=
                    static_cast<int>(instance->SEQ[sequence].size()) *
                    active
            );
            expressions.continuity -= active;
            ++result.variables;
            result.constraints += 2;

            for (int agent = 0; agent < instance->A; ++agent) {
                IloIntExpr count(environment);
                int terms = 0;
                for (int service : instance->SEQ[sequence]) {
                    if (!exists[agent][service])
                        continue;
                    count += IloPresenceOf(
                        environment, alternatives[agent][service]
                    );
                    ++terms;
                }
                if (terms == 0) {
                    count.end();
                    continue;
                }
                IloBoolVar used(
                    environment,
                    cpName(
                        "sequence_agent",
                        agent,
                        static_cast<int>(sequence)
                    ).c_str()
                );
                model.add(count >= used);
                model.add(count <= terms * used);
                expressions.continuity += used;
                ++result.variables;
                result.constraints += 2;
                count.end();
            }
            activeCount.end();
        }

        result.constraints += addCPBoundsAndObjective(
            environment, model, request, expressions
        );
        result.buildSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - buildStarted
        ).count();
        const double solveBudget =
            request.timeoutSeconds - result.buildSeconds;
        if (solveBudget <= 0) {
            result.status = COMMERCIAL_TIMEOUT;
            result.message =
                "cumulative timeout exhausted during model build";
            model.end();
            return result;
        }

        IloCP cp(model);
        ofstream log;
        configureCP(
            cp, config, solveBudget, log, environment
        );
        auto solveStarted = chrono::steady_clock::now();
        const bool solved = cp.solve();
        result.solveSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - solveStarted
        ).count();
        finishCPResult(cp, solved, result);
        if (solved) {
            for (int agent = 0; agent < instance->A; ++agent) {
                for (int service = 0; service < instance->S; ++service) {
                    if (!exists[agent][service] ||
                        !cp.isPresent(alternatives[agent][service]))
                        continue;
                    result.assignments.push_back(make_tuple(
                        agent,
                        service,
                        static_cast<int>(
                            cp.getStart(alternatives[agent][service])
                        )
                    ));
                }
            }
        }
        cp.end();
        model.end();
        return result;
    }

public:
    explicit CplexCPBackend(const HCORAPBackendConfig &config)
        : config(config), environment() {
        if (!config.parameterFile.empty()) {
            throw runtime_error(
                "--parameter-file is only supported by MIP backends; "
                "configure CP Optimizer through the locked CLI options"
            );
        }
        if (config.formulation != "cp-t" &&
            config.formulation != "cp-i")
            throw runtime_error(
                "cplex-cp formulation must be cp-t or cp-i"
            );
        IloCP cp(environment);
        productVersion = cp.getVersion();
        cp.end();
    }

    ~CplexCPBackend() { environment.end(); }

    string name() const { return "cplex-cp"; }
    string formulation() const { return config.formulation; }
    string version() const { return productVersion; }

    HCORAPStageResult solve(const HCORAPStageRequest &request) {
        try {
            return config.formulation == "cp-t"
                ? solveCPT(request) : solveCPI(request);
        } catch (const IloException &error) {
            HCORAPStageResult result;
            result.status = COMMERCIAL_ERROR;
            result.message =
                string("CP Optimizer exception: ") + error.getMessage();
            return result;
        } catch (const exception &error) {
            HCORAPStageResult result;
            result.status = COMMERCIAL_ERROR;
            result.message = error.what();
            return result;
        }
    }
};

}

bool hcorapCplexCompiled() { return true; }

unique_ptr<HCORAPCommercialBackend> createCplexCPBackend(
    const HCORAPBackendConfig &config
) {
    try {
        return unique_ptr<HCORAPCommercialBackend>(
            new CplexCPBackend(config)
        );
    } catch (const IloException &error) {
        throw runtime_error(
            string("cannot initialize CP Optimizer: ") +
            error.getMessage()
        );
    }
}

#else

bool hcorapCplexCompiled() { return false; }

unique_ptr<HCORAPCommercialBackend> createCplexCPBackend(
    const HCORAPBackendConfig &
) {
    throw runtime_error(
        "CP Optimizer backend was not compiled; rebuild with CPLEX=1"
    );
}

#endif
