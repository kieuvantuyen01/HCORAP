#include "CommercialTypes.h"
#include "HCORAPMIPModel.h"

#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

#ifdef HCORAP_WITH_GUROBI
#include "gurobi_c++.h"
#endif

using namespace std;

#ifdef HCORAP_WITH_GUROBI

namespace {

class GurobiMIPBackend : public HCORAPCommercialBackend {
    HCORAPBackendConfig config;
    GRBEnv environment;

    static GRBLinExpr translate(
        const HCORAPLinearExpression &source,
        const vector<GRBVar> &variables
    ) {
        GRBLinExpr expression(source.constant);
        for (const HCORAPLinearTerm &term : source.terms)
            expression += static_cast<double>(term.coefficient) *
                variables[term.variable];
        return expression;
    }

public:
    explicit GurobiMIPBackend(const HCORAPBackendConfig &config)
        : config(config), environment(true) {
        if (!config.parameterFile.empty())
            environment.readParams(config.parameterFile);
        environment.set(GRB_IntParam_OutputFlag, 0);
        if (!config.solverLog.empty()) {
            environment.set(GRB_IntParam_OutputFlag, 1);
            environment.set(GRB_StringParam_LogFile, config.solverLog);
        }
        environment.start();
    }

    string name() const { return "gurobi-mip"; }
    string formulation() const { return "mip-e"; }
    string version() const {
        ostringstream output;
        output << GRB_VERSION_MAJOR << '.' << GRB_VERSION_MINOR
               << '.' << GRB_VERSION_TECHNICAL;
        return output.str();
    }

    HCORAPStageResult solve(const HCORAPStageRequest &request) {
        HCORAPStageResult result;
        try {
            auto buildStarted = chrono::steady_clock::now();
            HCORAPMIPModel source = buildHCORAPMIPModel(request);
            GRBModel model(environment);
            model.set(GRB_IntParam_Threads, config.threads);
            model.set(GRB_IntParam_Seed, config.seed);
            model.set(GRB_DoubleParam_MIPGap, config.mipGap);
            model.set(GRB_DoubleParam_MIPGapAbs, config.absoluteMipGap);
            model.set(GRB_DoubleParam_FeasibilityTol, 1e-6);
            model.set(GRB_DoubleParam_IntFeasTol, 1e-5);

            vector<GRBVar> variables;
            variables.reserve(source.variables.size());
            for (const HCORAPLinearVariable &variable : source.variables) {
                variables.push_back(model.addVar(
                    variable.lowerBound,
                    variable.upperBound,
                    0.0,
                    variable.type == HCORAP_LINEAR_BINARY
                        ? GRB_BINARY : GRB_INTEGER,
                    variable.name
                ));
            }
            for (const HCORAPLinearConstraint &constraint :
                 source.constraints) {
                GRBLinExpr expression = translate(
                    constraint.expression, variables
                );
                char sense = GRB_EQUAL;
                if (constraint.sense == HCORAP_LINEAR_LE)
                    sense = GRB_LESS_EQUAL;
                else if (constraint.sense == HCORAP_LINEAR_GE)
                    sense = GRB_GREATER_EQUAL;
                model.addConstr(
                    expression,
                    sense,
                    static_cast<double>(constraint.rightHandSide),
                    constraint.name
                );
            }
            model.setObjective(
                translate(source.objective, variables),
                source.maximize ? GRB_MAXIMIZE : GRB_MINIMIZE
            );
            model.update();

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
                return result;
            }
            model.set(GRB_DoubleParam_TimeLimit, solveBudget);

            auto solveStarted = chrono::steady_clock::now();
            model.optimize();
            result.solveSeconds = chrono::duration<double>(
                chrono::steady_clock::now() - solveStarted
            ).count();

            const int status = model.get(GRB_IntAttr_Status);
            const int solutionCount = model.get(GRB_IntAttr_SolCount);
            if (status == GRB_OPTIMAL) {
                result.status = COMMERCIAL_OPTIMUM;
            } else if (
                status == GRB_INFEASIBLE ||
                status == GRB_INF_OR_UNBD
            ) {
                result.status = COMMERCIAL_INFEASIBLE;
            } else if (status == GRB_TIME_LIMIT) {
                result.status = solutionCount > 0
                    ? COMMERCIAL_TIMEOUT_FEASIBLE : COMMERCIAL_TIMEOUT;
            } else {
                result.status = COMMERCIAL_ERROR;
                result.message = "Gurobi status " + to_string(status);
            }

            if (solutionCount > 0) {
                for (const auto &entry : source.assignmentVariables) {
                    if (variables[entry.second].get(GRB_DoubleAttr_X) > 0.5)
                        result.assignments.push_back(entry.first);
                }
                result.bestBound = model.get(GRB_DoubleAttr_ObjBound);
                result.relativeGap = model.get(GRB_DoubleAttr_MIPGap);
                result.hasBestBound = true;
                result.hasRelativeGap = true;
            } else if (
                status != GRB_INFEASIBLE &&
                status != GRB_INF_OR_UNBD
            ) {
                result.bestBound = model.get(GRB_DoubleAttr_ObjBound);
                result.hasBestBound = true;
            }
            result.explored = static_cast<long long>(
                model.get(GRB_DoubleAttr_NodeCount)
            );
        } catch (const GRBException &error) {
            result.status = COMMERCIAL_ERROR;
            ostringstream message;
            message << "Gurobi error " << error.getErrorCode()
                    << ": " << error.getMessage();
            result.message = message.str();
        } catch (const exception &error) {
            result.status = COMMERCIAL_ERROR;
            result.message = error.what();
        }
        return result;
    }
};

}

bool hcorapGurobiCompiled() { return true; }

unique_ptr<HCORAPCommercialBackend> createGurobiMIPBackend(
    const HCORAPBackendConfig &config
) {
    try {
        return unique_ptr<HCORAPCommercialBackend>(
            new GurobiMIPBackend(config)
        );
    } catch (const GRBException &error) {
        ostringstream message;
        message << "cannot initialize Gurobi (error "
                << error.getErrorCode() << "): "
                << error.getMessage();
        throw runtime_error(message.str());
    }
}

#else

bool hcorapGurobiCompiled() { return false; }

unique_ptr<HCORAPCommercialBackend> createGurobiMIPBackend(
    const HCORAPBackendConfig &
) {
    throw runtime_error(
        "Gurobi backend was not compiled; rebuild with GUROBI=1"
    );
}

#endif
