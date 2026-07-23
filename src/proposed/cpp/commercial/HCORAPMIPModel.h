#ifndef HCORAP_MIP_MODEL_H
#define HCORAP_MIP_MODEL_H

#include "CommercialTypes.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

enum HCORAPLinearVariableType {
    HCORAP_LINEAR_BINARY,
    HCORAP_LINEAR_INTEGER
};

enum HCORAPLinearConstraintSense {
    HCORAP_LINEAR_LE,
    HCORAP_LINEAR_EQ,
    HCORAP_LINEAR_GE
};

struct HCORAPLinearVariable {
    std::string name;
    int lowerBound;
    int upperBound;
    HCORAPLinearVariableType type;
};

struct HCORAPLinearTerm {
    int variable;
    long long coefficient;

    HCORAPLinearTerm();
    HCORAPLinearTerm(int variable, long long coefficient);
};

struct HCORAPLinearExpression {
    std::vector<HCORAPLinearTerm> terms;
    long long constant;

    HCORAPLinearExpression();
    void add(int variable, long long coefficient);
    void add(const HCORAPLinearExpression &other, long long multiplier = 1);
};

struct HCORAPLinearConstraint {
    std::string name;
    HCORAPLinearExpression expression;
    HCORAPLinearConstraintSense sense;
    long long rightHandSide;
};

struct HCORAPMIPModel {
    std::vector<HCORAPLinearVariable> variables;
    std::vector<HCORAPLinearConstraint> constraints;
    HCORAPLinearExpression objective;
    bool maximize;

    HCORAPLinearExpression coverage;
    HCORAPLinearExpression similarity;
    HCORAPLinearExpression continuity;
    HCORAPLinearExpression overtime;

    std::map<std::tuple<int, int, int>, int> assignmentVariables;
    std::vector<std::vector<int> > agentServiceVariables;
    std::vector<int> serviceVariables;
    std::vector<int> workloadVariables;
    std::vector<std::vector<int> > overtimeThresholdVariables;
    std::vector<int> sequenceActiveVariables;
    std::vector<std::vector<int> > sequenceAgentVariables;

    int addVariable(
        const std::string &name,
        int lowerBound,
        int upperBound,
        HCORAPLinearVariableType type
    );
    void addConstraint(
        const std::string &name,
        const HCORAPLinearExpression &expression,
        HCORAPLinearConstraintSense sense,
        long long rightHandSide
    );
};

HCORAPMIPModel buildHCORAPMIPModel(const HCORAPStageRequest &request);

long long evaluateHCORAPLinearExpression(
    const HCORAPLinearExpression &expression,
    const std::vector<int> &values
);

std::vector<std::string> validateHCORAPMIPSchedule(
    const HCORAPMIPModel &model,
    const HCORAPStageRequest &request,
    const std::vector<std::tuple<int, int, int> > &assignments,
    long long *objectiveValue
);

#endif
