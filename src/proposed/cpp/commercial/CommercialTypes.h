#ifndef HCORAP_COMMERCIAL_TYPES_H
#define HCORAP_COMMERCIAL_TYPES_H

#include "hcorap.h"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

enum HCORAPCommercialObjective {
    COMMERCIAL_WEIGHTED,
    COMMERCIAL_COVERAGE,
    COMMERCIAL_SIMILARITY,
    COMMERCIAL_CONTINUITY,
    COMMERCIAL_OVERTIME
};

enum HCORAPCommercialStatus {
    COMMERCIAL_OPTIMUM,
    COMMERCIAL_INFEASIBLE,
    COMMERCIAL_TIMEOUT_FEASIBLE,
    COMMERCIAL_TIMEOUT,
    COMMERCIAL_ERROR
};

struct HCORAPCommercialBounds {
    int minCoverage;
    int minSimilarity;
    int maxContinuity;
    int maxOvertime;

    HCORAPCommercialBounds();
};

struct HCORAPCommercialMetrics {
    bool valid;
    int coverage;
    int similarity;
    int continuity;
    int overtime;
    int overtimeCost;
    std::vector<int> workload;
    std::vector<std::tuple<int, int, int> > assignments;
    std::vector<std::string> violations;

    HCORAPCommercialMetrics();
};

struct HCORAPBackendConfig {
    std::string backend;
    std::string formulation;
    std::string parameterFile;
    std::string solverLog;
    int threads;
    int seed;
    double mipGap;
    double absoluteMipGap;
    long long enumerationLimit;

    HCORAPBackendConfig();
};

struct HCORAPStageRequest {
    const HCORAP *instance;
    HCORAPCommercialObjective objective;
    HCORAPCommercialBounds bounds;
    bool fullCoverage;
    int continuityWeight;
    int overtimeWeight;
    double timeoutSeconds;
    int stageIndex;

    HCORAPStageRequest();
};

struct HCORAPStageResult {
    HCORAPCommercialStatus status;
    std::vector<std::tuple<int, int, int> > assignments;
    double buildSeconds;
    double solveSeconds;
    double bestBound;
    double relativeGap;
    bool hasBestBound;
    bool hasRelativeGap;
    int variables;
    int constraints;
    long long explored;
    std::string message;

    HCORAPStageResult();
};

class HCORAPCommercialBackend {
public:
    virtual ~HCORAPCommercialBackend() {}
    virtual std::string name() const = 0;
    virtual std::string formulation() const = 0;
    virtual std::string version() const = 0;
    virtual HCORAPStageResult solve(const HCORAPStageRequest &request) = 0;
};

const char *hcorapCommercialObjectiveName(HCORAPCommercialObjective objective);
const char *hcorapCommercialObjectiveSense(HCORAPCommercialObjective objective);
const char *hcorapCommercialStatusName(HCORAPCommercialStatus status);

int hcorapCommercialObjectiveValue(
    HCORAPCommercialObjective objective,
    const HCORAPCommercialMetrics &metrics,
    const HCORAP *instance,
    int continuityWeight,
    int overtimeWeight
);

bool hcorapCommercialBoundsSatisfied(
    const HCORAPCommercialBounds &bounds,
    const HCORAPCommercialMetrics &metrics
);

std::vector<std::string> validateHCORAPInstance(const HCORAP *instance);

HCORAPCommercialMetrics verifyHCORAPAssignments(
    const HCORAP *instance,
    const std::vector<std::tuple<int, int, int> > &assignments,
    bool requireFullCoverage
);

bool hcorapGurobiCompiled();
bool hcorapCplexCompiled();

std::unique_ptr<HCORAPCommercialBackend> createGurobiMIPBackend(
    const HCORAPBackendConfig &config
);
std::unique_ptr<HCORAPCommercialBackend> createCplexMIPBackend(
    const HCORAPBackendConfig &config
);
std::unique_ptr<HCORAPCommercialBackend> createCplexCPBackend(
    const HCORAPBackendConfig &config
);
std::unique_ptr<HCORAPCommercialBackend> createReferenceBackend(
    const HCORAPBackendConfig &config
);

#endif
