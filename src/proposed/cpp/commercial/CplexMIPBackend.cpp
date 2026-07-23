#include "CommercialTypes.h"
#include "HCORAPMIPModel.h"

#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef HCORAP_WITH_CPLEX
#include <ilcplex/ilocplex.h>
ILOSTLBEGIN
#endif

using namespace std;

#ifdef HCORAP_WITH_CPLEX

namespace {

class CplexMIPBackend : public HCORAPCommercialBackend {
    HCORAPBackendConfig config;
    IloEnv environment;

    IloExpr translate(
        const HCORAPLinearExpression &source,
        const IloNumVarArray &variables
    ) const {
        IloExpr expression(environment);
        expression += static_cast<IloNum>(source.constant);
        for (const HCORAPLinearTerm &term : source.terms) {
            expression += static_cast<IloNum>(term.coefficient) *
                variables[term.variable];
        }
        return expression;
    }

public:
    explicit CplexMIPBackend(const HCORAPBackendConfig &config)
        : config(config), environment() {}

    ~CplexMIPBackend() { environment.end(); }

    string name() const { return "cplex-mip"; }
    string formulation() const { return "mip-e"; }
    string version() const { return CPX_VERSION; }

    HCORAPStageResult solve(const HCORAPStageRequest &request) {
        HCORAPStageResult result;
        try {
            auto buildStarted = chrono::steady_clock::now();
            HCORAPMIPModel source = buildHCORAPMIPModel(request);
            IloModel model(environment);
            IloNumVarArray variables(environment);
            for (const HCORAPLinearVariable &variable : source.variables) {
                variables.add(IloNumVar(
                    environment,
                    variable.lowerBound,
                    variable.upperBound,
                    variable.type == HCORAP_LINEAR_BINARY
                        ? ILOBOOL : ILOINT,
                    variable.name.c_str()
                ));
            }
            for (const HCORAPLinearConstraint &constraint :
                 source.constraints) {
                IloExpr expression = translate(
                    constraint.expression, variables
                );
                if (constraint.sense == HCORAP_LINEAR_LE) {
                    model.add(
                        expression <=
                        static_cast<IloNum>(constraint.rightHandSide)
                    );
                } else if (constraint.sense == HCORAP_LINEAR_GE) {
                    model.add(
                        expression >=
                        static_cast<IloNum>(constraint.rightHandSide)
                    );
                } else {
                    model.add(
                        expression ==
                        static_cast<IloNum>(constraint.rightHandSide)
                    );
                }
                expression.end();
            }
            IloExpr objective = translate(source.objective, variables);
            if (source.maximize)
                model.add(IloMaximize(environment, objective));
            else
                model.add(IloMinimize(environment, objective));
            objective.end();

            IloCplex cplex(model);
            if (!config.parameterFile.empty())
                cplex.readParam(config.parameterFile.c_str());
            cplex.setParam(
                IloCplex::Param::DetTimeLimit,
                cplex.getDefault(IloCplex::Param::DetTimeLimit)
            );
            cplex.setParam(
                IloCplex::Param::ClockType,
                2  // Wall-clock time, as required by the experiment protocol.
            );
            cplex.setParam(IloCplex::Param::Threads, config.threads);
            cplex.setParam(IloCplex::Param::RandomSeed, config.seed);
            cplex.setParam(
                IloCplex::Param::MIP::Tolerances::MIPGap,
                config.mipGap
            );
            cplex.setParam(
                IloCplex::Param::MIP::Tolerances::AbsMIPGap,
                config.absoluteMipGap
            );
            cplex.setParam(
                IloCplex::Param::Simplex::Tolerances::Feasibility,
                1e-6
            );
            cplex.setParam(
                IloCplex::Param::MIP::Tolerances::Integrality,
                1e-5
            );

            ofstream log;
            if (config.solverLog.empty()) {
                cplex.setOut(environment.getNullStream());
                cplex.setWarning(environment.getNullStream());
            } else {
                log.open(config.solverLog.c_str(), ios::out | ios::app);
                if (!log)
                    throw runtime_error(
                        "cannot open CPLEX log: " + config.solverLog
                    );
                cplex.setOut(log);
                cplex.setWarning(log);
            }

            result.variables = static_cast<int>(source.variables.size());
            result.constraints = static_cast<int>(source.constraints.size());
            result.buildSeconds = chrono::duration<double>(
                chrono::steady_clock::now() - buildStarted
            ).count();
            const double solveBudget =
                request.timeoutSeconds - result.buildSeconds;
            if (solveBudget <= 0) {
                result.status = COMMERCIAL_TIMEOUT;
                result.message =
                    "cumulative timeout exhausted during model build";
                cplex.end();
                model.end();
                return result;
            }
            cplex.setParam(IloCplex::Param::TimeLimit, solveBudget);

            auto solveStarted = chrono::steady_clock::now();
            const bool solved = cplex.solve();
            result.solveSeconds = chrono::duration<double>(
                chrono::steady_clock::now() - solveStarted
            ).count();
            const IloAlgorithm::Status status = cplex.getStatus();
            const IloCplex::CplexStatus detailedStatus =
                cplex.getCplexStatus();
            const bool hasIncumbent =
                solved ||
                status == IloAlgorithm::Feasible ||
                status == IloAlgorithm::Optimal;
            if (status == IloAlgorithm::Optimal) {
                result.status = COMMERCIAL_OPTIMUM;
            } else if (
                status == IloAlgorithm::Infeasible ||
                detailedStatus == IloCplex::InfOrUnbd
            ) {
                result.status = COMMERCIAL_INFEASIBLE;
            } else if (detailedStatus == IloCplex::AbortTimeLim) {
                result.status = hasIncumbent
                    ? COMMERCIAL_TIMEOUT_FEASIBLE
                    : COMMERCIAL_TIMEOUT;
            } else {
                result.status = COMMERCIAL_ERROR;
                ostringstream message;
                if (detailedStatus == IloCplex::AbortDetTimeLim) {
                    message
                        << "CPLEX reached an unexpected deterministic-time "
                        << "limit; the experiment protocol requires the "
                        << "wall-clock TimeLimit";
                } else if (detailedStatus == IloCplex::AbortUser) {
                    message << "CPLEX solve was aborted by the user";
                } else {
                    message << "CPLEX stopped for a reason other than the "
                            << "wall-clock TimeLimit";
                }
                message << " (status " << status
                        << ", CPLEX substatus " << detailedStatus << ")";
                result.message = message.str();
            }

            if (hasIncumbent) {
                for (const auto &entry : source.assignmentVariables) {
                    if (cplex.getValue(variables[entry.second]) > 0.5)
                        result.assignments.push_back(entry.first);
                }
                result.bestBound = cplex.getBestObjValue();
                result.relativeGap = cplex.getMIPRelativeGap();
                result.hasBestBound = true;
                result.hasRelativeGap = true;
            } else if (
                status != IloAlgorithm::Infeasible &&
                detailedStatus != IloCplex::InfOrUnbd
            ) {
                result.bestBound = cplex.getBestObjValue();
                result.hasBestBound = true;
            }
            result.explored = static_cast<long long>(cplex.getNnodes());
            cplex.end();
            model.end();
        } catch (const IloException &error) {
            result.status = COMMERCIAL_ERROR;
            result.message = string("CPLEX exception: ") + error.getMessage();
        } catch (const exception &error) {
            result.status = COMMERCIAL_ERROR;
            result.message = error.what();
        }
        return result;
    }
};

}

unique_ptr<HCORAPCommercialBackend> createCplexMIPBackend(
    const HCORAPBackendConfig &config
) {
    try {
        return unique_ptr<HCORAPCommercialBackend>(
            new CplexMIPBackend(config)
        );
    } catch (const IloException &error) {
        throw runtime_error(
            string("cannot initialize CPLEX: ") + error.getMessage()
        );
    }
}

#else

unique_ptr<HCORAPCommercialBackend> createCplexMIPBackend(
    const HCORAPBackendConfig &
) {
    throw runtime_error(
        "CPLEX backend was not compiled; rebuild with CPLEX=1"
    );
}

#endif
