#include "HCORAPMIPModel.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace std;

HCORAPLinearTerm::HCORAPLinearTerm() : variable(-1), coefficient(0) {}

HCORAPLinearTerm::HCORAPLinearTerm(int variable, long long coefficient)
    : variable(variable), coefficient(coefficient) {}

HCORAPLinearExpression::HCORAPLinearExpression() : constant(0) {}

void HCORAPLinearExpression::add(int variable, long long coefficient) {
    if (coefficient != 0)
        terms.push_back(HCORAPLinearTerm(variable, coefficient));
}

void HCORAPLinearExpression::add(
    const HCORAPLinearExpression &other,
    long long multiplier
) {
    constant += multiplier * other.constant;
    for (const HCORAPLinearTerm &term : other.terms)
        add(term.variable, multiplier * term.coefficient);
}

int HCORAPMIPModel::addVariable(
    const string &name,
    int lowerBound,
    int upperBound,
    HCORAPLinearVariableType type
) {
    HCORAPLinearVariable variable;
    variable.name = name;
    variable.lowerBound = lowerBound;
    variable.upperBound = upperBound;
    variable.type = type;
    variables.push_back(variable);
    return static_cast<int>(variables.size()) - 1;
}

void HCORAPMIPModel::addConstraint(
    const string &name,
    const HCORAPLinearExpression &expression,
    HCORAPLinearConstraintSense sense,
    long long rightHandSide
) {
    HCORAPLinearConstraint constraint;
    constraint.name = name;
    constraint.expression = expression;
    constraint.sense = sense;
    constraint.rightHandSide = rightHandSide;
    constraints.push_back(constraint);
}

static string indexedName(
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

static bool isCandidate(
    const HCORAP *instance,
    int agent,
    int service,
    int slot
) {
    return instance->r[agent][service] > 0 &&
        instance->TSA[agent][slot] &&
        instance->TSS[service][slot];
}

static void addExpressionBound(
    HCORAPMIPModel &model,
    const string &name,
    const HCORAPLinearExpression &expression,
    HCORAPLinearConstraintSense sense,
    long long value
) {
    model.addConstraint(
        name, expression, sense, value - expression.constant
    );
    model.constraints.back().expression.constant = 0;
}

HCORAPMIPModel buildHCORAPMIPModel(const HCORAPStageRequest &request) {
    if (request.instance == NULL)
        throw runtime_error("cannot build MIP model for a null instance");
    const HCORAP *instance = request.instance;
    HCORAPMIPModel model;

    vector<vector<vector<int> > > x(
        instance->A,
        vector<vector<int> >(
            instance->S, vector<int>(instance->TS, -1)
        )
    );
    vector<vector<int> > y(
        instance->A, vector<int>(instance->S, -1)
    );
    vector<int> z(instance->S, -1);
    vector<int> workload(instance->A, -1);
    vector<vector<int> > sequenceAgent(
        instance->A, vector<int>(instance->SEQ.size(), -1)
    );
    vector<int> sequenceActive(instance->SEQ.size(), -1);
    vector<vector<int> > overtimeThreshold(instance->A);

    for (int service = 0; service < instance->S; ++service) {
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (!isCandidate(instance, agent, service, slot))
                    continue;
                int variable = model.addVariable(
                    indexedName("x", agent, service, slot),
                    0, 1, HCORAP_LINEAR_BINARY
                );
                x[agent][service][slot] = variable;
                model.assignmentVariables[
                    make_tuple(agent, service, slot)
                ] = variable;
            }
        }
    }

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            bool hasCandidate = false;
            for (int slot = 0; slot < instance->TS; ++slot)
                hasCandidate = hasCandidate || x[agent][service][slot] >= 0;
            if (hasCandidate) {
                y[agent][service] = model.addVariable(
                    indexedName("y", agent, service),
                    0, 1, HCORAP_LINEAR_BINARY
                );
                model.similarity.add(
                    y[agent][service], instance->r[agent][service]
                );
            }
        }
    }

    for (int service = 0; service < instance->S; ++service) {
        z[service] = model.addVariable(
            indexedName("z", service),
            request.fullCoverage ? 1 : 0,
            1,
            HCORAP_LINEAR_BINARY
        );
        model.coverage.add(z[service], 1);

        HCORAPLinearExpression link;
        link.add(z[service], 1);
        for (int agent = 0; agent < instance->A; ++agent) {
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (x[agent][service][slot] >= 0)
                    link.add(x[agent][service][slot], -1);
            }
        }
        model.addConstraint(
            indexedName("service_assignment", service),
            link, HCORAP_LINEAR_EQ, 0
        );
    }

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            if (y[agent][service] < 0)
                continue;
            HCORAPLinearExpression link;
            link.add(y[agent][service], 1);
            for (int slot = 0; slot < instance->TS; ++slot) {
                if (x[agent][service][slot] >= 0)
                    link.add(x[agent][service][slot], -1);
            }
            model.addConstraint(
                indexedName("agent_service", agent, service),
                link, HCORAP_LINEAR_EQ, 0
            );
        }
    }

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            HCORAPLinearExpression conflict;
            for (int service = 0; service < instance->S; ++service) {
                if (x[agent][service][slot] >= 0)
                    conflict.add(x[agent][service][slot], 1);
            }
            if (conflict.terms.size() > 1) {
                model.addConstraint(
                    indexedName("agent_slot", agent, slot),
                    conflict, HCORAP_LINEAR_LE, 1
                );
            }
        }
    }

    for (size_t user = 0; user < instance->SU.size(); ++user) {
        for (int slot = 0; slot < instance->TS; ++slot) {
            HCORAPLinearExpression conflict;
            for (int service : instance->SU[user]) {
                for (int agent = 0; agent < instance->A; ++agent) {
                    if (x[agent][service][slot] >= 0)
                        conflict.add(x[agent][service][slot], 1);
                }
            }
            if (conflict.terms.size() > 1) {
                model.addConstraint(
                    indexedName(
                        "user_slot", static_cast<int>(user), slot
                    ),
                    conflict, HCORAP_LINEAR_LE, 1
                );
            }
        }
    }

    for (int agent = 0; agent < instance->A; ++agent) {
        const int capacity = instance->HN[agent] + instance->HE[agent];
        workload[agent] = model.addVariable(
            indexedName("workload", agent),
            0, capacity, HCORAP_LINEAR_INTEGER
        );
        HCORAPLinearExpression link;
        link.add(workload[agent], 1);
        for (int service = 0; service < instance->S; ++service) {
            if (y[agent][service] >= 0)
                link.add(y[agent][service], -1);
        }
        model.addConstraint(
            indexedName("workload_definition", agent),
            link, HCORAP_LINEAR_EQ, 0
        );

        for (int unit = 1; unit <= instance->HE[agent]; ++unit) {
            const int threshold = instance->HN[agent] + unit;
            int overtime = model.addVariable(
                indexedName("overtime", agent, unit),
                0, 1, HCORAP_LINEAR_BINARY
            );
            overtimeThreshold[agent].push_back(overtime);
            model.overtime.add(overtime, 1);

            HCORAPLinearExpression lower;
            lower.add(workload[agent], 1);
            lower.add(overtime, -threshold);
            model.addConstraint(
                indexedName("overtime_lower", agent, unit),
                lower, HCORAP_LINEAR_GE, 0
            );

            HCORAPLinearExpression upper;
            upper.add(workload[agent], 1);
            upper.add(
                overtime,
                -(capacity - threshold + 1)
            );
            model.addConstraint(
                indexedName("overtime_upper", agent, unit),
                upper, HCORAP_LINEAR_LE, threshold - 1
            );
        }
    }

    for (size_t sequence = 0; sequence < instance->SEQ.size(); ++sequence) {
        sequenceActive[sequence] = model.addVariable(
            indexedName("sequence_active", static_cast<int>(sequence)),
            0, 1, HCORAP_LINEAR_BINARY
        );
        HCORAPLinearExpression activeInputs;
        for (int service : instance->SEQ[sequence]) {
            activeInputs.add(z[service], 1);
            HCORAPLinearExpression implication;
            implication.add(sequenceActive[sequence], 1);
            implication.add(z[service], -1);
            model.addConstraint(
                indexedName(
                    "sequence_active_lower",
                    static_cast<int>(sequence),
                    service
                ),
                implication, HCORAP_LINEAR_GE, 0
            );
        }
        HCORAPLinearExpression activeUpper;
        activeUpper.add(sequenceActive[sequence], 1);
        activeUpper.add(activeInputs, -1);
        model.addConstraint(
            indexedName(
                "sequence_active_upper", static_cast<int>(sequence)
            ),
            activeUpper, HCORAP_LINEAR_LE, 0
        );
        model.continuity.add(sequenceActive[sequence], -1);

        for (int agent = 0; agent < instance->A; ++agent) {
            vector<int> inputs;
            for (int service : instance->SEQ[sequence]) {
                if (y[agent][service] >= 0)
                    inputs.push_back(y[agent][service]);
            }
            if (inputs.empty())
                continue;
            int used = model.addVariable(
                indexedName(
                    "sequence_agent", agent, static_cast<int>(sequence)
                ),
                0, 1, HCORAP_LINEAR_BINARY
            );
            sequenceAgent[agent][sequence] = used;
            model.continuity.add(used, 1);

            for (int input : inputs) {
                HCORAPLinearExpression lower;
                lower.add(used, 1);
                lower.add(input, -1);
                model.addConstraint(
                    indexedName(
                        "sequence_agent_lower",
                        agent,
                        static_cast<int>(sequence),
                        input
                    ),
                    lower, HCORAP_LINEAR_GE, 0
                );
            }
            HCORAPLinearExpression upper;
            upper.add(used, 1);
            for (int input : inputs)
                upper.add(input, -1);
            model.addConstraint(
                indexedName(
                    "sequence_agent_upper",
                    agent,
                    static_cast<int>(sequence)
                ),
                upper, HCORAP_LINEAR_LE, 0
            );
        }
    }

    if (request.bounds.minCoverage >= 0) {
        addExpressionBound(
            model, "bound_coverage", model.coverage,
            HCORAP_LINEAR_GE, request.bounds.minCoverage
        );
    }
    if (request.bounds.minSimilarity >= 0) {
        addExpressionBound(
            model, "bound_similarity", model.similarity,
            HCORAP_LINEAR_GE, request.bounds.minSimilarity
        );
    }
    if (request.bounds.maxContinuity >= 0) {
        addExpressionBound(
            model, "bound_continuity", model.continuity,
            HCORAP_LINEAR_LE, request.bounds.maxContinuity
        );
    }
    if (request.bounds.maxOvertime >= 0) {
        addExpressionBound(
            model, "bound_overtime", model.overtime,
            HCORAP_LINEAR_LE, request.bounds.maxOvertime
        );
    }

    switch (request.objective) {
        case COMMERCIAL_COVERAGE:
            model.objective = model.coverage;
            model.maximize = true;
            break;
        case COMMERCIAL_SIMILARITY:
            model.objective = model.similarity;
            model.maximize = true;
            break;
        case COMMERCIAL_CONTINUITY:
            model.objective = model.continuity;
            model.maximize = false;
            break;
        case COMMERCIAL_OVERTIME:
            model.objective = model.overtime;
            model.maximize = false;
            break;
        case COMMERCIAL_WEIGHTED:
        default:
            model.objective.add(model.similarity, 1);
            model.objective.add(
                model.continuity, -request.continuityWeight
            );
            model.objective.add(
                model.overtime,
                -static_cast<long long>(request.overtimeWeight) *
                    abs(instance->P)
            );
            model.maximize = true;
            break;
    }
    model.agentServiceVariables = y;
    model.serviceVariables = z;
    model.workloadVariables = workload;
    model.overtimeThresholdVariables = overtimeThreshold;
    model.sequenceActiveVariables = sequenceActive;
    model.sequenceAgentVariables = sequenceAgent;
    return model;
}

long long evaluateHCORAPLinearExpression(
    const HCORAPLinearExpression &expression,
    const vector<int> &values
) {
    long long result = expression.constant;
    for (const HCORAPLinearTerm &term : expression.terms) {
        if (term.variable < 0 ||
            term.variable >= static_cast<int>(values.size()))
            throw runtime_error("linear expression variable out of range");
        result += term.coefficient * values[term.variable];
    }
    return result;
}

vector<string> validateHCORAPMIPSchedule(
    const HCORAPMIPModel &model,
    const HCORAPStageRequest &request,
    const vector<tuple<int, int, int> > &assignments,
    long long *objectiveValue
) {
    vector<string> violations;
    const HCORAP *instance = request.instance;
    if (instance == NULL) {
        violations.push_back("null instance");
        return violations;
    }
    vector<int> values(model.variables.size(), 0);
    vector<int> covered(instance->S, 0);
    vector<int> workloads(instance->A, 0);
    vector<vector<bool> > assigned(
        instance->A, vector<bool>(instance->S, false)
    );

    for (const tuple<int, int, int> &assignment : assignments) {
        map<tuple<int, int, int>, int>::const_iterator found =
            model.assignmentVariables.find(assignment);
        if (found == model.assignmentVariables.end()) {
            violations.push_back(
                "schedule contains a triple without an x variable"
            );
            continue;
        }
        values[found->second] = 1;
        const int agent = get<0>(assignment);
        const int service = get<1>(assignment);
        if (agent < 0 || agent >= instance->A ||
            service < 0 || service >= instance->S)
            continue;
        covered[service] = 1;
        ++workloads[agent];
        assigned[agent][service] = true;
    }

    for (int agent = 0; agent < instance->A; ++agent) {
        for (int service = 0; service < instance->S; ++service) {
            const int variable =
                model.agentServiceVariables[agent][service];
            if (variable >= 0)
                values[variable] = assigned[agent][service] ? 1 : 0;
        }
        values[model.workloadVariables[agent]] = workloads[agent];
        for (int unit = 1;
             unit <= instance->HE[agent]; ++unit) {
            values[
                model.overtimeThresholdVariables[agent][unit - 1]
            ] = workloads[agent] >= instance->HN[agent] + unit ? 1 : 0;
        }
    }
    for (int service = 0; service < instance->S; ++service)
        values[model.serviceVariables[service]] = covered[service];

    for (size_t sequence = 0;
         sequence < instance->SEQ.size(); ++sequence) {
        bool active = false;
        for (int service : instance->SEQ[sequence])
            active = active || covered[service] != 0;
        values[model.sequenceActiveVariables[sequence]] =
            active ? 1 : 0;
        for (int agent = 0; agent < instance->A; ++agent) {
            const int variable =
                model.sequenceAgentVariables[agent][sequence];
            if (variable < 0)
                continue;
            bool used = false;
            for (int service : instance->SEQ[sequence])
                used = used || assigned[agent][service];
            values[variable] = used ? 1 : 0;
        }
    }

    for (size_t index = 0; index < model.variables.size(); ++index) {
        if (values[index] < model.variables[index].lowerBound ||
            values[index] > model.variables[index].upperBound) {
            violations.push_back(
                "variable bound violated: " +
                model.variables[index].name
            );
        }
    }
    for (const HCORAPLinearConstraint &constraint : model.constraints) {
        const long long left = evaluateHCORAPLinearExpression(
            constraint.expression, values
        );
        bool satisfied = false;
        if (constraint.sense == HCORAP_LINEAR_LE)
            satisfied = left <= constraint.rightHandSide;
        else if (constraint.sense == HCORAP_LINEAR_GE)
            satisfied = left >= constraint.rightHandSide;
        else
            satisfied = left == constraint.rightHandSide;
        if (!satisfied)
            violations.push_back(
                "linear constraint violated: " + constraint.name
            );
    }
    if (objectiveValue != NULL) {
        *objectiveValue = evaluateHCORAPLinearExpression(
            model.objective, values
        );
    }
    return violations;
}
