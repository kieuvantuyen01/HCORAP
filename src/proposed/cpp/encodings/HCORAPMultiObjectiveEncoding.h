#ifndef HCORAP_MULTI_OBJECTIVE_ENCODING_H
#define HCORAP_MULTI_OBJECTIVE_ENCODING_H

#include "CardinalityNetwork.h"
#include "ImpliedConstraints.h"
#include "SymmetryBreaking.h"
#include "encoding.h"
#include "hcorap.h"

#include <tuple>
#include <vector>

enum HCORAPObjectiveKind {
    HCORAP_WEIGHTED,
    HCORAP_COVERAGE,
    HCORAP_SIMILARITY,
    HCORAP_CONTINUITY,
    HCORAP_OVERTIME
};

struct HCORAPObjectiveBounds {
    int minCoverage;
    int minSimilarity;
    int maxContinuity;
    int maxOvertime;

    HCORAPObjectiveBounds();
};

struct HCORAPSolutionMetrics {
    bool valid;
    int coverage;
    int similarity;
    int continuity;
    int overtime;
    int overtimeCost;
    std::vector<int> workload;
    std::vector<std::tuple<int, int, int> > assignments;

    HCORAPSolutionMetrics();
};

class HCORAPMultiObjectiveEncoding : public Encoding {
    HCORAP *instance;
    HCORAPObjectiveKind objective;
    HCORAPObjectiveBounds bounds;
    bool fullCoverage;
    int continuityWeight;
    int overtimeWeight;
    HCORAPCardinalityEncoding cardinalityEncoding;
    HCORAPImpliedConfig impliedConfig;
    HCORAPSymmetryBreaking symmetryBreaking;

    std::vector<std::vector<std::vector<literal> > > x;
    std::vector<std::vector<literal> > y;
    std::vector<std::vector<literal> > serviceSlot;
    std::vector<std::vector<literal> > userUsedSlot;
    std::vector<literal> performed;
    std::vector<std::vector<literal> > sequenceAgent;
    std::vector<literal> sequenceActive;
    std::vector<std::vector<literal> > overtimeThreshold;
    std::vector<bool> model;

    bool literalValue(const literal &value) const;
    void addAtMostOne(SMTFormula *formula, const std::vector<literal> &values);
    void addCardinalityAtMost(
        SMTFormula *formula,
        const std::vector<literal> &values,
        int bound
    );
    void addCardinalityExactly(
        SMTFormula *formula,
        const std::vector<literal> &values,
        int target
    );
    void addEqualCardinality(
        SMTFormula *formula,
        const std::vector<literal> &left,
        const std::vector<literal> &right
    );
    void addProjectedServiceAssignments(SMTFormula *formula);
    void addServiceSlotVariables(SMTFormula *formula);
    void addUserSlotConstraints(SMTFormula *formula);
    void addSlotCapacityConstraints(SMTFormula *formula);
    void addValuePrecedence(
        SMTFormula *formula,
        const std::vector<literal> &earlier,
        const std::vector<literal> &later
    );
    void addSlotSymmetryBreaking(SMTFormula *formula);
    void addServiceSymmetryBreaking(SMTFormula *formula);
    void addAgentSymmetryBreaking(SMTFormula *formula);
    bool hasSlotSymmetry() const;
    int slotMatchingCapacity(int slot) const;
    int effectiveWorkloadCapacity(int agent) const;
    void addPBAtLeast(
        SMTFormula *formula,
        const std::vector<int> &weights,
        const std::vector<literal> &values,
        int lowerBound
    );
    void addBounds(SMTFormula *formula);
    void addObjective(SMTFormula *formula);

public:
    HCORAPMultiObjectiveEncoding(
        HCORAP *instance,
        HCORAPObjectiveKind objective,
        bool fullCoverage,
        int continuityWeight,
        int overtimeWeight,
        HCORAPCardinalityEncoding cardinalityEncoding,
        HCORAPImpliedConfig impliedConfig,
        HCORAPSymmetryBreaking symmetryBreaking,
        const HCORAPObjectiveBounds &bounds
    );

    SMTFormula *encode(int lb = INT_MIN, int ub = INT_MAX);
    void setBooleanModel(const std::vector<bool> &values);
    HCORAPSolutionMetrics evaluateModel() const;
    int objectiveValue(const HCORAPSolutionMetrics &metrics) const;
};

#endif
