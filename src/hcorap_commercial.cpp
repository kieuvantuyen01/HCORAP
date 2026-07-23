#include "parser.h"
#include "CommercialTypes.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace {

struct Options {
    string instancePath;
    string backend;
    string formulation;
    string method;
    string outputPath;
    string delta;
    double timeoutSeconds;
    bool fullCoverage;
    bool printAssignments;
    bool listBackends;
    int continuityWeight;
    int overtimeWeight;
    HCORAPBackendConfig backendConfig;

    Options()
        : method("weighted"), delta("0.05"), timeoutSeconds(3600.0),
          fullCoverage(true), printAssignments(false), listBackends(false),
          continuityWeight(1), overtimeWeight(1) {}
};

struct StageRecord {
    string name;
    HCORAPCommercialObjective objective;
    HCORAPStageResult result;
    bool hasIncumbent;
    int incumbentValue;
    double verificationSeconds;

    StageRecord()
        : objective(COMMERCIAL_WEIGHTED), hasIncumbent(false),
          incumbentValue(0), verificationSeconds(0.0) {}
};

struct RunState {
    HCORAPCommercialStatus status;
    vector<StageRecord> stages;
    HCORAPCommercialMetrics metrics;
    bool hasMetrics;
    int metricsStageIndex;
    int solverCalls;
    string error;
    int similarityReferenceOptimum;
    int similarityLowerBound;

    RunState()
        : status(COMMERCIAL_ERROR), hasMetrics(false),
          metricsStageIndex(-1), solverCalls(0),
          similarityReferenceOptimum(-1), similarityLowerBound(-1) {}
};

static void usage(const char *program) {
    cerr
        << "Usage: " << program << " INSTANCE --backend BACKEND [options]\n"
        << "       " << program << " --list-backends\n"
        << "Backends:\n"
        << "  gurobi-mip | cplex-mip | cplex-cp | reference-enumerator\n"
        << "Formulations:\n"
        << "  mip-e (MIP), cp-t or cp-i (CP Optimizer)\n"
        << "Optimization:\n"
        << "  --method weighted|lex-continuity|lex-overtime|epsilon\n"
        << "  --soft-coverage           maximize and fix coverage first\n"
        << "  --wc INTEGER              continuity weight in weighted mode\n"
        << "  --wo INTEGER              overtime multiplier in weighted mode\n"
        << "  --delta DECIMAL           similarity loss budget for epsilon\n"
        << "Reproducibility and limits:\n"
        << "  --timeout SECONDS         cumulative limit across all stages\n"
        << "  --threads INTEGER         deterministic default: 1\n"
        << "  --seed INTEGER            random seed, default: 0\n"
        << "  --mip-gap DECIMAL         relative MIP gap, default: 0\n"
        << "  --absolute-mip-gap VALUE  absolute MIP gap, default: 0\n"
        << "  --parameter-file FILE     Gurobi/CPLEX MIP parameter file\n"
        << "  --solver-log FILE         append native solver log\n"
        << "  --enumeration-limit N     reference backend leaf limit\n"
        << "Output:\n"
        << "  --output FILE             write JSON result to FILE\n"
        << "  --print-assignments       include assignment triples in JSON\n";
}

static string requireValue(int argc, char **argv, int &index) {
    if (index + 1 >= argc)
        throw runtime_error(string("missing value after ") + argv[index]);
    return argv[++index];
}

static pair<long long, long long> parseDecimalFraction(const string &text) {
    if (text.empty())
        throw runtime_error("delta must be a decimal in [0,1]");
    const size_t dot = text.find('.');
    if (dot != string::npos && text.find('.', dot + 1) != string::npos)
        throw runtime_error("delta must be a decimal in [0,1]");
    string whole = dot == string::npos ? text : text.substr(0, dot);
    const string fraction =
        dot == string::npos ? "" : text.substr(dot + 1);
    if (whole.empty() && fraction.empty())
        throw runtime_error("delta must contain at least one digit");
    if (whole.empty())
        whole = "0";
    if (fraction.size() > 9)
        throw runtime_error("delta supports at most 9 decimal places");
    for (char character : whole) {
        if (character < '0' || character > '9')
            throw runtime_error("delta must be a decimal in [0,1]");
    }
    for (char character : fraction) {
        if (character < '0' || character > '9')
            throw runtime_error("delta must be a decimal in [0,1]");
    }
    long long scale = 1;
    for (size_t index = 0; index < fraction.size(); ++index)
        scale *= 10;
    long long numerator = stoll(whole) * scale;
    if (!fraction.empty())
        numerator += stoll(fraction);
    if (numerator < 0 || numerator > scale)
        throw runtime_error("delta must be in [0,1]");
    return make_pair(numerator, scale);
}

static int similarityThreshold(int optimum, const string &delta) {
    const pair<long long, long long> fraction =
        parseDecimalFraction(delta);
    const long long numerator =
        (fraction.second - fraction.first) * optimum;
    return static_cast<int>(
        (numerator + fraction.second - 1) / fraction.second
    );
}

static Options parseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            exit(0);
        } else if (argument == "--list-backends") {
            options.listBackends = true;
        } else if (argument == "--backend") {
            options.backend = requireValue(argc, argv, index);
        } else if (argument == "--formulation") {
            options.formulation = requireValue(argc, argv, index);
        } else if (argument == "--method") {
            options.method = requireValue(argc, argv, index);
        } else if (argument == "--timeout") {
            options.timeoutSeconds =
                stod(requireValue(argc, argv, index));
        } else if (argument == "--threads") {
            options.backendConfig.threads =
                stoi(requireValue(argc, argv, index));
        } else if (argument == "--seed") {
            options.backendConfig.seed =
                stoi(requireValue(argc, argv, index));
        } else if (argument == "--mip-gap") {
            options.backendConfig.mipGap =
                stod(requireValue(argc, argv, index));
        } else if (argument == "--absolute-mip-gap") {
            options.backendConfig.absoluteMipGap =
                stod(requireValue(argc, argv, index));
        } else if (argument == "--parameter-file") {
            options.backendConfig.parameterFile =
                requireValue(argc, argv, index);
        } else if (argument == "--solver-log") {
            options.backendConfig.solverLog =
                requireValue(argc, argv, index);
        } else if (argument == "--enumeration-limit") {
            options.backendConfig.enumerationLimit =
                stoll(requireValue(argc, argv, index));
        } else if (argument == "--wc") {
            options.continuityWeight =
                stoi(requireValue(argc, argv, index));
        } else if (argument == "--wo") {
            options.overtimeWeight =
                stoi(requireValue(argc, argv, index));
        } else if (argument == "--delta") {
            options.delta = requireValue(argc, argv, index);
        } else if (argument == "--soft-coverage") {
            options.fullCoverage = false;
        } else if (argument == "--output") {
            options.outputPath = requireValue(argc, argv, index);
        } else if (argument == "--print-assignments") {
            options.printAssignments = true;
        } else if (!argument.empty() && argument[0] == '-') {
            throw runtime_error("unknown option: " + argument);
        } else if (options.instancePath.empty()) {
            options.instancePath = argument;
        } else {
            throw runtime_error(
                "unexpected positional argument: " + argument
            );
        }
    }

    if (options.listBackends)
        return options;
    if (options.instancePath.empty())
        throw runtime_error("an instance path is required");
    if (options.backend.empty())
        throw runtime_error("--backend is required");
    if (options.backend != "gurobi-mip" &&
        options.backend != "cplex-mip" &&
        options.backend != "cplex-cp" &&
        options.backend != "reference-enumerator")
        throw runtime_error("unsupported backend: " + options.backend);

    if (options.formulation.empty()) {
        if (options.backend == "cplex-cp")
            options.formulation = "cp-t";
        else if (options.backend == "reference-enumerator")
            options.formulation = "direct-schedule-enumeration";
        else
            options.formulation = "mip-e";
    }
    if ((options.backend == "gurobi-mip" ||
         options.backend == "cplex-mip") &&
        options.formulation != "mip-e")
        throw runtime_error("MIP backends require --formulation mip-e");
    if (options.backend == "cplex-cp" &&
        options.formulation != "cp-t" &&
        options.formulation != "cp-i")
        throw runtime_error(
            "cplex-cp requires --formulation cp-t or cp-i"
        );
    if (options.backend == "reference-enumerator" &&
        options.formulation != "direct-schedule-enumeration")
        throw runtime_error(
            "reference-enumerator uses direct-schedule-enumeration"
        );
    if (options.method != "weighted" &&
        options.method != "lex-continuity" &&
        options.method != "lex-overtime" &&
        options.method != "epsilon")
        throw runtime_error("unsupported method: " + options.method);
    if (!isfinite(options.timeoutSeconds) ||
        options.timeoutSeconds <= 0)
        throw runtime_error("timeout must be finite and positive");
    if (options.backendConfig.threads <= 0)
        throw runtime_error("threads must be positive");
    if (options.backendConfig.seed < 0)
        throw runtime_error("seed must be non-negative");
    if (!isfinite(options.backendConfig.mipGap) ||
        options.backendConfig.mipGap < 0 ||
        options.backendConfig.mipGap > 1)
        throw runtime_error("mip-gap must be finite and in [0,1]");
    if (!isfinite(options.backendConfig.absoluteMipGap) ||
        options.backendConfig.absoluteMipGap < 0)
        throw runtime_error(
            "absolute-mip-gap must be finite and non-negative"
        );
    if (options.backendConfig.enumerationLimit <= 0)
        throw runtime_error("enumeration-limit must be positive");
    if (options.continuityWeight < 0 ||
        options.overtimeWeight < 0)
        throw runtime_error("wc and wo must be non-negative");
    if (options.method == "epsilon")
        parseDecimalFraction(options.delta);
    if (!options.backendConfig.parameterFile.empty() &&
        options.backend != "gurobi-mip" &&
        options.backend != "cplex-mip")
        throw runtime_error(
            "--parameter-file applies only to Gurobi/CPLEX MIP"
        );
    if (!options.backendConfig.solverLog.empty() &&
        options.backend == "reference-enumerator")
        throw runtime_error(
            "--solver-log is not supported by reference-enumerator"
        );
    if (options.backend == "cplex-cp" &&
        (options.backendConfig.mipGap != 0 ||
         options.backendConfig.absoluteMipGap != 0))
        throw runtime_error("MIP gap options do not apply to cplex-cp");
    if (options.backendConfig.mipGap != 0 ||
        options.backendConfig.absoluteMipGap != 0)
        throw runtime_error(
            "certified commercial comparisons require zero MIP gaps; "
            "a positive-gap incumbent is not labeled OPTIMUM"
        );

    options.backendConfig.backend = options.backend;
    options.backendConfig.formulation = options.formulation;
    return options;
}

static unique_ptr<HCORAPCommercialBackend> createBackend(
    const HCORAPBackendConfig &config
) {
    if (config.backend == "gurobi-mip")
        return createGurobiMIPBackend(config);
    if (config.backend == "cplex-mip")
        return createCplexMIPBackend(config);
    if (config.backend == "cplex-cp")
        return createCplexCPBackend(config);
    return createReferenceBackend(config);
}

static void validateObjectiveMagnitude(
    const HCORAP *instance,
    const Options &options
) {
    long long maximumSimilarity = 0;
    for (int service = 0; service < instance->S; ++service) {
        int maximumReward = 0;
        for (int agent = 0; agent < instance->A; ++agent)
            maximumReward = max(
                maximumReward, instance->r[agent][service]
            );
        maximumSimilarity += maximumReward;
    }
    long long maximumContinuity = 0;
    for (const vector<int> &sequence : instance->SEQ) {
        maximumContinuity += max(
            0,
            min(
                instance->A,
                static_cast<int>(sequence.size())
            ) - 1
        );
    }
    long long maximumOvertime = 0;
    for (int extra : instance->HE)
        maximumOvertime += extra;
    const long long overtimeCoefficient =
        static_cast<long long>(options.overtimeWeight) *
        abs(instance->P);
    const long long maximumPenalty =
        static_cast<long long>(options.continuityWeight) *
            maximumContinuity +
        overtimeCoefficient * maximumOvertime;
    const long long integerMaximum = numeric_limits<int>::max();
    if (maximumSimilarity > integerMaximum ||
        maximumContinuity > integerMaximum ||
        maximumOvertime > integerMaximum ||
        static_cast<long long>(abs(instance->P)) *
            maximumOvertime > integerMaximum ||
        overtimeCoefficient > integerMaximum ||
        maximumPenalty > integerMaximum)
        throw runtime_error(
            "objective magnitude exceeds the exact integer range "
            "supported by the shared driver"
        );
}

static void updateBound(
    HCORAPCommercialBounds &bounds,
    HCORAPCommercialObjective objective,
    const HCORAPCommercialMetrics &metrics
) {
    switch (objective) {
        case COMMERCIAL_COVERAGE:
            bounds.minCoverage = metrics.coverage;
            break;
        case COMMERCIAL_SIMILARITY:
            bounds.minSimilarity = metrics.similarity;
            break;
        case COMMERCIAL_CONTINUITY:
            bounds.maxContinuity = metrics.continuity;
            break;
        case COMMERCIAL_OVERTIME:
            bounds.maxOvertime = metrics.overtime;
            break;
        case COMMERCIAL_WEIGHTED:
        default:
            break;
    }
}

static double elapsedSince(
    const chrono::steady_clock::time_point &started
) {
    return chrono::duration<double>(
        chrono::steady_clock::now() - started
    ).count();
}

static HCORAPCommercialStatus executeStage(
    HCORAPCommercialBackend &backend,
    const HCORAP *instance,
    const Options &options,
    HCORAPCommercialBounds &bounds,
    HCORAPCommercialObjective objective,
    const string &stageName,
    bool preserveOptimum,
    const chrono::steady_clock::time_point &overallStarted,
    RunState &state
) {
    StageRecord record;
    record.name = stageName;
    record.objective = objective;
    const double remaining =
        options.timeoutSeconds - elapsedSince(overallStarted);
    if (remaining <= 0) {
        record.result.status = COMMERCIAL_TIMEOUT;
        record.result.message =
            "cumulative timeout exhausted before stage construction";
        state.stages.push_back(record);
        return COMMERCIAL_TIMEOUT;
    }

    HCORAPStageRequest request;
    request.instance = instance;
    request.objective = objective;
    request.bounds = bounds;
    request.fullCoverage = options.fullCoverage;
    request.continuityWeight = options.continuityWeight;
    request.overtimeWeight = options.overtimeWeight;
    request.timeoutSeconds = remaining;
    request.stageIndex = static_cast<int>(state.stages.size());
    ++state.solverCalls;
    record.result = backend.solve(request);

    const bool feasible =
        record.result.status == COMMERCIAL_OPTIMUM ||
        record.result.status == COMMERCIAL_TIMEOUT_FEASIBLE;
    if (feasible) {
        const chrono::steady_clock::time_point verificationStarted =
            chrono::steady_clock::now();
        HCORAPCommercialMetrics verified = verifyHCORAPAssignments(
            instance, record.result.assignments, options.fullCoverage
        );
        if (!verified.valid) {
            record.result.status = COMMERCIAL_ERROR;
            record.result.message =
                "independent verifier rejected the incumbent";
            if (!verified.violations.empty())
                record.result.message += ": " + verified.violations.front();
        } else if (!hcorapCommercialBoundsSatisfied(bounds, verified)) {
            record.result.status = COMMERCIAL_ERROR;
            record.result.message =
                "independent verifier rejected an inherited objective bound";
        } else {
            record.hasIncumbent = true;
            record.incumbentValue = hcorapCommercialObjectiveValue(
                objective,
                verified,
                instance,
                options.continuityWeight,
                options.overtimeWeight
            );
            state.metrics = verified;
            state.hasMetrics = true;
            state.metricsStageIndex =
                static_cast<int>(state.stages.size());
            if (record.result.status == COMMERCIAL_OPTIMUM &&
                record.result.hasBestBound &&
                fabs(
                    record.result.bestBound -
                    static_cast<double>(record.incumbentValue)
                ) > 1e-4) {
                record.result.status = COMMERCIAL_ERROR;
                record.result.message =
                    "optimal incumbent and reported best bound disagree";
            }
        }
        record.verificationSeconds =
            elapsedSince(verificationStarted);
    }

    const HCORAPCommercialStatus status = record.result.status;
    if (status == COMMERCIAL_OPTIMUM && preserveOptimum)
        updateBound(bounds, objective, state.metrics);
    if (status == COMMERCIAL_ERROR && !record.result.message.empty())
        state.error = record.result.message;
    state.stages.push_back(record);
    return status;
}

static RunState solvePolicy(
    HCORAPCommercialBackend &backend,
    const HCORAP *instance,
    const Options &options,
    const chrono::steady_clock::time_point &overallStarted
) {
    RunState state;
    HCORAPCommercialBounds bounds;
    HCORAPCommercialStatus status = COMMERCIAL_OPTIMUM;

    if (!options.fullCoverage) {
        status = executeStage(
            backend, instance, options, bounds, COMMERCIAL_COVERAGE,
            "coverage", true, overallStarted, state
        );
    }

    if (status == COMMERCIAL_OPTIMUM && options.method == "weighted") {
        status = executeStage(
            backend, instance, options, bounds, COMMERCIAL_WEIGHTED,
            "weighted_score", false, overallStarted, state
        );
    } else if (
        status == COMMERCIAL_OPTIMUM &&
        (options.method == "lex-continuity" ||
         options.method == "lex-overtime")
    ) {
        vector<HCORAPCommercialObjective> order;
        if (options.method == "lex-continuity") {
            order.push_back(COMMERCIAL_CONTINUITY);
            order.push_back(COMMERCIAL_SIMILARITY);
            order.push_back(COMMERCIAL_OVERTIME);
        } else {
            order.push_back(COMMERCIAL_OVERTIME);
            order.push_back(COMMERCIAL_CONTINUITY);
            order.push_back(COMMERCIAL_SIMILARITY);
        }
        for (HCORAPCommercialObjective objective : order) {
            status = executeStage(
                backend, instance, options, bounds, objective,
                hcorapCommercialObjectiveName(objective), true,
                overallStarted, state
            );
            if (status != COMMERCIAL_OPTIMUM)
                break;
        }
    } else if (
        status == COMMERCIAL_OPTIMUM && options.method == "epsilon"
    ) {
        status = executeStage(
            backend, instance, options, bounds, COMMERCIAL_SIMILARITY,
            "similarity_reference", false, overallStarted, state
        );
        if (status == COMMERCIAL_OPTIMUM) {
            state.similarityReferenceOptimum =
                state.metrics.similarity;
            state.similarityLowerBound = similarityThreshold(
                state.similarityReferenceOptimum, options.delta
            );
            bounds.minSimilarity = state.similarityLowerBound;
        }
        if (status == COMMERCIAL_OPTIMUM) {
            status = executeStage(
                backend, instance, options, bounds,
                COMMERCIAL_CONTINUITY, "continuity", true,
                overallStarted, state
            );
        }
        if (status == COMMERCIAL_OPTIMUM) {
            status = executeStage(
                backend, instance, options, bounds,
                COMMERCIAL_OVERTIME, "overtime", true,
                overallStarted, state
            );
        }
        if (status == COMMERCIAL_OPTIMUM) {
            status = executeStage(
                backend, instance, options, bounds,
                COMMERCIAL_SIMILARITY, "similarity_tiebreak", false,
                overallStarted, state
            );
        }
    }

    state.status = status;
    return state;
}

static void jsonString(ostream &output, const string &value) {
    output << '"';
    for (char character : value) {
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character; break;
        }
    }
    output << '"';
}

static const char *objectiveMode(const string &method) {
    if (method == "weighted")
        return "weighted";
    if (method == "epsilon")
        return "epsilon-constraint";
    return "lexicographic";
}

static void writeNullableNumber(
    ostream &output, bool available, double value
) {
    if (available && isfinite(value))
        output << setprecision(12) << value;
    else
        output << "null";
}

static void writeResult(
    ostream &output,
    const Options &options,
    const HCORAPCommercialBackend &backend,
    const HCORAP *instance,
    const RunState &state,
    double elapsedSeconds
) {
    output << "{\n  \"schema_version\": 1,\n  \"status\": ";
    jsonString(output, hcorapCommercialStatusName(state.status));
    output << ",\n  \"instance\": ";
    jsonString(output, options.instancePath);
    output << ",\n  \"language\": \"C++\",\n  \"backend\": ";
    jsonString(output, backend.name());
    output << ",\n  \"formulation\": ";
    jsonString(output, backend.formulation());
    output << ",\n  \"solver_version\": ";
    jsonString(output, backend.version());
    output << ",\n  \"method\": ";
    jsonString(output, options.method);
    output << ",\n  \"objective_mode\": ";
    jsonString(output, objectiveMode(options.method));
    output << ",\n  \"timing_scope\": "
           << "\"parse+build+solve+verify; cumulative across stages\",\n"
           << "  \"timeout_seconds\": " << options.timeoutSeconds << ",\n"
           << "  \"elapsed_seconds\": " << setprecision(12)
           << elapsedSeconds << ",\n"
           << "  \"solver_calls\": " << state.solverCalls << ",\n"
           << "  \"full_coverage\": "
           << (options.fullCoverage ? "true" : "false") << ",\n"
           << "  \"threads\": " << options.backendConfig.threads << ",\n"
           << "  \"seed\": " << options.backendConfig.seed << ",\n"
           << "  \"mip_gap\": " << options.backendConfig.mipGap << ",\n"
           << "  \"absolute_mip_gap\": "
           << options.backendConfig.absoluteMipGap << ",\n"
           << "  \"parameter_file\": ";
    if (options.backendConfig.parameterFile.empty())
        output << "null";
    else
        jsonString(output, options.backendConfig.parameterFile);
    output << ",\n  \"solver_log\": ";
    if (options.backendConfig.solverLog.empty())
        output << "null";
    else
        jsonString(output, options.backendConfig.solverLog);
    const bool mipBackend =
        backend.name() == "gurobi-mip" ||
        backend.name() == "cplex-mip";
    const bool cpBackend = backend.name() == "cplex-cp";
    output << ",\n  \"mip_feasibility_tolerance\": ";
    if (mipBackend)
        output << "1e-6";
    else
        output << "null";
    output << ",\n  \"mip_integrality_tolerance\": ";
    if (mipBackend)
        output << "1e-5";
    else
        output << "null";
    output << ",\n  \"cp_absolute_optimality_tolerance\": ";
    if (cpBackend)
        output << '0';
    else
        output << "null";
    output << ",\n  \"cp_relative_optimality_tolerance\": ";
    if (cpBackend)
        output << '0';
    else
        output << "null";
    output << ",\n"
           << "  \"continuity_weight\": "
           << options.continuityWeight << ",\n"
           << "  \"overtime_weight\": "
           << options.overtimeWeight << ",\n"
           << "  \"overtime_penalty_per_hour\": "
           << abs(instance->P) << ",\n"
           << "  \"delta\": ";
    jsonString(output, options.delta);
    output << ",\n  \"similarity_reference_optimum\": ";
    if (state.similarityReferenceOptimum >= 0)
        output << state.similarityReferenceOptimum;
    else
        output << "null";
    output << ",\n  \"similarity_lower_bound\": ";
    if (state.similarityLowerBound >= 0)
        output << state.similarityLowerBound;
    else
        output << "null";

    output << ",\n  \"stages\": [";
    for (size_t index = 0; index < state.stages.size(); ++index) {
        const StageRecord &stage = state.stages[index];
        if (index)
            output << ',';
        output << "\n    {\"index\": " << index << ", \"name\": ";
        jsonString(output, stage.name);
        output << ", \"objective\": ";
        jsonString(
            output, hcorapCommercialObjectiveName(stage.objective)
        );
        output << ", \"sense\": ";
        jsonString(
            output, hcorapCommercialObjectiveSense(stage.objective)
        );
        output << ", \"status\": ";
        jsonString(
            output, hcorapCommercialStatusName(stage.result.status)
        );
        output << ", \"incumbent\": ";
        if (stage.hasIncumbent)
            output << stage.incumbentValue;
        else
            output << "null";
        output << ", \"best_bound\": ";
        writeNullableNumber(
            output, stage.result.hasBestBound, stage.result.bestBound
        );
        output << ", \"relative_gap\": ";
        writeNullableNumber(
            output, stage.result.hasRelativeGap,
            stage.result.relativeGap
        );
        output << ", \"build_seconds\": "
               << stage.result.buildSeconds
               << ", \"solve_seconds\": "
               << stage.result.solveSeconds
               << ", \"verification_seconds\": "
               << stage.verificationSeconds
               << ", \"variables\": " << stage.result.variables
               << ", \"constraints\": " << stage.result.constraints
               << ", \"search_nodes_or_branches\": "
               << stage.result.explored
               << ", \"message\": ";
        if (stage.result.message.empty())
            output << "null";
        else
            jsonString(output, stage.result.message);
        output << '}';
    }
    if (!state.stages.empty())
        output << '\n';
    output << "  ],\n  \"incumbent_stage_index\": ";
    if (state.metricsStageIndex >= 0)
        output << state.metricsStageIndex;
    else
        output << "null";
    output << ",\n  \"error\": ";
    if (state.error.empty())
        output << "null";
    else
        jsonString(output, state.error);
    output << ",\n  \"metrics\": ";
    if (!state.hasMetrics) {
        output << "null";
    } else {
        const int weightedScore = hcorapCommercialObjectiveValue(
            COMMERCIAL_WEIGHTED,
            state.metrics,
            instance,
            options.continuityWeight,
            options.overtimeWeight
        );
        output << "{\"coverage\": " << state.metrics.coverage
               << ", \"similarity\": " << state.metrics.similarity
               << ", \"continuity\": " << state.metrics.continuity
               << ", \"overtime\": " << state.metrics.overtime
               << ", \"overtime_cost\": " << state.metrics.overtimeCost
               << ", \"weighted_reference_score\": "
               << weightedScore
               << ", \"verified\": "
               << (state.metrics.valid ? "true" : "false")
               << ", \"workload\": [";
        for (size_t index = 0;
             index < state.metrics.workload.size(); ++index) {
            if (index)
                output << ',';
            output << state.metrics.workload[index];
        }
        output << "]}";
    }
    if (options.printAssignments && state.hasMetrics) {
        output << ",\n  \"assignments\": [";
        for (size_t index = 0;
             index < state.metrics.assignments.size(); ++index) {
            if (index)
                output << ',';
            output << '['
                   << get<0>(state.metrics.assignments[index]) << ','
                   << get<1>(state.metrics.assignments[index]) << ','
                   << get<2>(state.metrics.assignments[index]) << ']';
        }
        output << ']';
    }
    output << "\n}\n";
}

static void writeBackends(ostream &output) {
    output
        << "{\n  \"backends\": [\n"
        << "    {\"name\": \"gurobi-mip\", "
        << "\"formulations\": [\"mip-e\"], \"compiled\": "
        << (hcorapGurobiCompiled() ? "true" : "false") << "},\n"
        << "    {\"name\": \"cplex-mip\", "
        << "\"formulations\": [\"mip-e\"], \"compiled\": "
        << (hcorapCplexCompiled() ? "true" : "false") << "},\n"
        << "    {\"name\": \"cplex-cp\", "
        << "\"formulations\": [\"cp-t\", \"cp-i\"], "
        << "\"compiled\": "
        << (hcorapCplexCompiled() ? "true" : "false") << "},\n"
        << "    {\"name\": \"reference-enumerator\", "
        << "\"formulations\": [\"direct-schedule-enumeration\"], "
        << "\"compiled\": true}\n"
        << "  ]\n}\n";
}

}

int main(int argc, char **argv) {
    try {
        const Options options = parseOptions(argc, argv);
        if (options.listBackends) {
            writeBackends(cout);
            return 0;
        }

        const chrono::steady_clock::time_point started =
            chrono::steady_clock::now();
        unique_ptr<HCORAP> instance(
            parser::parseHCORAP(options.instancePath)
        );
        const vector<string> violations =
            validateHCORAPInstance(instance.get());
        if (!violations.empty())
            throw runtime_error(
                "invalid HCORAP instance: " + violations.front()
            );
        validateObjectiveMagnitude(instance.get(), options);

        unique_ptr<HCORAPCommercialBackend> backend =
            createBackend(options.backendConfig);
        const RunState state = solvePolicy(
            *backend, instance.get(), options, started
        );
        const double elapsedSeconds = elapsedSince(started);

        if (options.outputPath.empty()) {
            writeResult(
                cout, options, *backend, instance.get(),
                state, elapsedSeconds
            );
        } else {
            ofstream output(options.outputPath.c_str());
            if (!output)
                throw runtime_error(
                    "cannot open output file: " + options.outputPath
                );
            writeResult(
                output, options, *backend, instance.get(),
                state, elapsedSeconds
            );
        }
        return state.status == COMMERCIAL_OPTIMUM ? 0 : 2;
    } catch (const exception &error) {
        cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
