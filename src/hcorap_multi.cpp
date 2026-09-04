#include "parser.h"
#include "HCORAPMultiObjectiveEncoding.h"
#include "dimacsfileencoder.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std;

struct Options {
    string instancePath;
    string method;
    string solverPath;
    string outputPath;
    string delta;
    double timeoutSeconds;
    double solverShutdownGraceSeconds;
    int continuityWeight;
    int overtimeWeight;
    HCORAPCardinalityEncoding cardinalityEncoding;
    HCORAPImpliedConfig impliedConfig;
    HCORAPSymmetryBreaking symmetryBreaking;
    bool fullCoverage;
    bool encodeOnly;
    bool keepFiles;
    bool printAssignments;
    bool alignEvalMaxSATTargetTime;
    bool stage3IncumbentBound;

    Options()
        : method("weighted"), solverPath("EvalMaxSAT_bin"), delta("0.05"),
          timeoutSeconds(3600.0), solverShutdownGraceSeconds(5.0),
          continuityWeight(1), overtimeWeight(1),
          cardinalityEncoding(HCORAP_SORTING_NETWORK),
          impliedConfig(HCORAP_IMPLIED_NONE),
          symmetryBreaking(HCORAP_SYMMETRY_NONE),
          fullCoverage(true), encodeOnly(false), keepFiles(false),
          printAssignments(false), alignEvalMaxSATTargetTime(false),
          stage3IncumbentBound(false) {}
};

enum ExternalStatus {
    EXTERNAL_OPTIMUM,
    EXTERNAL_UNSAT,
    EXTERNAL_TIMEOUT,
    EXTERNAL_TIMEOUT_FEASIBLE,
    EXTERNAL_ERROR
};

struct ExternalResult {
    ExternalStatus status;
    vector<bool> model;
    double elapsedSeconds;
    double shutdownGraceSeconds;
    bool solverLaunched;
    string message;

    ExternalResult()
        : status(EXTERNAL_ERROR), elapsedSeconds(0.0),
          shutdownGraceSeconds(0.0), solverLaunched(false) {}
};

struct StageRecord {
    string name;
    string sense;
    string status;
    int optimum;
    int incumbent;
    bool hasOptimum;
    bool hasIncumbent;
    double encodeSeconds;
    double solveSeconds;
    double solverTargetSeconds;
    bool hasSolverTarget;
    bool solverLaunched;
    double shutdownGraceSeconds;
    int variables;
    int hardClauses;
    int softClauses;
    int continuityObjectiveWeight;
    int overtimeObjectiveWeight;
    string message;

    StageRecord()
        : optimum(0), incumbent(0), hasOptimum(false), hasIncumbent(false),
          encodeSeconds(0.0), solveSeconds(0.0), solverTargetSeconds(0.0),
          hasSolverTarget(false), solverLaunched(false),
          shutdownGraceSeconds(0.0), variables(0), hardClauses(0),
          softClauses(0), continuityObjectiveWeight(0),
          overtimeObjectiveWeight(0) {}
};

struct RunState {
    string status;
    vector<StageRecord> stages;
    HCORAPSolutionMetrics metrics;
    string error;
    int solverCalls;
    int provedStages;
    int incumbentStageIndex;
    bool hasIncumbent;
    int similarityReferenceOptimum;
    int similarityLowerBound;

    RunState()
        : solverCalls(0), provedStages(0), incumbentStageIndex(-1),
          hasIncumbent(false), similarityReferenceOptimum(-1),
          similarityLowerBound(-1) {}
};

static pair<long long, long long> parseDecimalFraction(const string &text);
static string statusName(ExternalStatus status);

static void usage(const char *program) {
    cerr
        << "Usage: " << program << " INSTANCE [options]\n"
        << "  --method weighted|lex-continuity|lex-cos|lex-overtime|"
           "lex-cos-one-shot|lex-overtime-one-shot|epsilon\n"
        << "  --solver PATH             external MaxSAT solver with s/v output\n"
        << "  --timeout SECONDS         cumulative encode+solve timeout\n"
        << "  --solver-shutdown-grace SECONDS\n"
        << "                            time to collect a model after SIGTERM\n"
        << "  --align-evalmaxsat-tct    pass the remaining budget via --TCT\n"
        << "  --stage3-incumbent-bound  constrain final similarity by Stage 2 incumbent\n"
        << "  --wc INTEGER              continuity weight (weighted)\n"
        << "  --wo INTEGER              overtime multiplier (weighted)\n"
        << "  --cardinality-encoding sorting-network|totalizer\n"
        << "  --implied-constraints none|user-slots|slot-capacity|both|both-plus\n"
        << "  --symmetry-breaking none|slots|services|slot-service|all\n"
        << "  --delta DECIMAL           similarity loss budget\n"
        << "  --soft-coverage           maximize/fix coverage first\n"
        << "  --encode-only             output weighted WCNF and do not solve\n"
        << "  --output FILE             write JSON result to FILE\n"
        << "  --keep-files              retain temporary WCNF/solver output\n"
        << "  --print-assignments       include assignments in JSON\n";
}

static string requireValue(int argc, char **argv, int &index) {
    if (index + 1 >= argc)
        throw runtime_error(string("missing value after ") + argv[index]);
    return argv[++index];
}

static Options parseOptions(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            exit(0);
        } else if (argument == "--method") {
            options.method = requireValue(argc, argv, index);
        } else if (argument == "--solver") {
            options.solverPath = requireValue(argc, argv, index);
        } else if (argument == "--timeout") {
            options.timeoutSeconds = stod(requireValue(argc, argv, index));
        } else if (argument == "--solver-shutdown-grace") {
            options.solverShutdownGraceSeconds =
                stod(requireValue(argc, argv, index));
        } else if (argument == "--align-evalmaxsat-tct") {
            options.alignEvalMaxSATTargetTime = true;
        } else if (argument == "--stage3-incumbent-bound") {
            options.stage3IncumbentBound = true;
        } else if (argument == "--wc") {
            options.continuityWeight = stoi(requireValue(argc, argv, index));
        } else if (argument == "--wo") {
            options.overtimeWeight = stoi(requireValue(argc, argv, index));
        } else if (argument == "--cardinality-encoding") {
            options.cardinalityEncoding = parseHCORAPCardinalityEncoding(
                requireValue(argc, argv, index)
            );
        } else if (argument == "--implied-constraints") {
            options.impliedConfig = parseHCORAPImpliedConfig(
                requireValue(argc, argv, index)
            );
        } else if (argument == "--symmetry-breaking") {
            options.symmetryBreaking = parseHCORAPSymmetryBreaking(
                requireValue(argc, argv, index)
            );
        } else if (argument == "--delta") {
            options.delta = requireValue(argc, argv, index);
        } else if (argument == "--output") {
            options.outputPath = requireValue(argc, argv, index);
        } else if (argument == "--soft-coverage") {
            options.fullCoverage = false;
        } else if (argument == "--encode-only") {
            options.encodeOnly = true;
        } else if (argument == "--keep-files") {
            options.keepFiles = true;
        } else if (argument == "--print-assignments") {
            options.printAssignments = true;
        } else if (!argument.empty() && argument[0] == '-') {
            throw runtime_error("unknown option: " + argument);
        } else if (options.instancePath.empty()) {
            options.instancePath = argument;
        } else {
            throw runtime_error("unexpected positional argument: " + argument);
        }
    }
    if (options.instancePath.empty())
        throw runtime_error("an instance path is required");
    if (!isfinite(options.timeoutSeconds) || options.timeoutSeconds <= 0)
        throw runtime_error("timeout must be positive");
    if (!isfinite(options.solverShutdownGraceSeconds) ||
        options.solverShutdownGraceSeconds < 0 ||
        options.solverShutdownGraceSeconds > 60)
        throw runtime_error("solver shutdown grace must lie in [0,60]");
    if (options.continuityWeight < 0 || options.overtimeWeight < 0)
        throw runtime_error("wc and wo must be non-negative");
    if (options.method != "weighted" && options.method != "lex-continuity" &&
        options.method != "lex-cos" && options.method != "lex-overtime" &&
        options.method != "lex-cos-one-shot" &&
        options.method != "lex-overtime-one-shot" &&
        options.method != "epsilon")
        throw runtime_error("unsupported method: " + options.method);
    if (options.stage3IncumbentBound &&
        options.method != "lex-cos" && options.method != "lex-overtime")
        throw runtime_error(
            "--stage3-incumbent-bound requires lex-cos or lex-overtime"
        );
    if (options.method == "epsilon")
        parseDecimalFraction(options.delta);
    if (options.encodeOnly && (options.method != "weighted" || !options.fullCoverage))
        throw runtime_error(
            "--encode-only currently represents full-coverage weighted mode"
        );
    return options;
}

static pair<long long, long long> parseDecimalFraction(const string &text) {
    if (text.empty())
        throw runtime_error("delta must be a decimal in [0,1]");
    size_t dot = text.find('.');
    if (dot != string::npos && text.find('.', dot + 1) != string::npos)
        throw runtime_error("delta must be a decimal in [0,1]");
    string whole = dot == string::npos ? text : text.substr(0, dot);
    string fraction = dot == string::npos ? "" : text.substr(dot + 1);
    if (whole.empty() && fraction.empty())
        throw runtime_error("delta must contain at least one digit");
    if (whole.empty())
        whole = "0";
    if (fraction.size() > 9)
        throw runtime_error("delta supports at most 9 decimal places");
    for (char character : whole) {
        if (!isdigit(static_cast<unsigned char>(character)))
            throw runtime_error("delta must be a decimal in [0,1]");
    }
    for (char character : fraction) {
        if (!isdigit(static_cast<unsigned char>(character)))
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
    pair<long long, long long> parsed = parseDecimalFraction(delta);
    long long numerator = (parsed.second - parsed.first) * optimum;
    return static_cast<int>((numerator + parsed.second - 1) / parsed.second);
}

static int greatestCommonDivisor(int left, int right) {
    while (right != 0) {
        int remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

static string trim(const string &value) {
    size_t start = value.find_first_not_of(" \t\r\n");
    if (start == string::npos)
        return "";
    size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

static ExternalResult parseSolverOutput(
    const string &path, int variables, double elapsedSeconds
) {
    ExternalResult result;
    result.elapsedSeconds = elapsedSeconds;
    result.model.assign(variables + 1, false);
    ifstream input(path.c_str());
    if (!input)
        throw runtime_error("cannot read solver output: " + path);

    bool optimum = false;
    bool unsat = false;
    bool satisfiable = false;
    bool modelFound = false;
    int binaryIndex = 1;
    string line;
    while (getline(input, line)) {
        string stripped = trim(line);
        if (stripped.size() >= 2 && stripped[0] == 's' && stripped[1] == ' ') {
            if (stripped.find("OPTIMUM") != string::npos)
                optimum = true;
            if (stripped.find("UNSATISFIABLE") != string::npos)
                unsat = true;
            else if (stripped.find("SATISFIABLE") != string::npos)
                satisfiable = true;
        }
        if (stripped.size() < 2 || stripped[0] != 'v' || stripped[1] != ' ')
            continue;
        string payload = trim(stripped.substr(2));
        bool bitString = !payload.empty();
        for (char character : payload) {
            if (character != '0' && character != '1') {
                bitString = false;
                break;
            }
        }
        if (bitString) {
            for (char character : payload) {
                if (binaryIndex <= variables)
                    result.model[binaryIndex] = character == '1';
                ++binaryIndex;
            }
            modelFound = true;
        } else {
            istringstream values(payload);
            int literalValue;
            while (values >> literalValue) {
                if (literalValue == 0)
                    break;
                int variable = abs(literalValue);
                if (variable <= variables)
                    result.model[variable] = literalValue > 0;
            }
            modelFound = true;
        }
    }

    if (unsat) {
        result.status = EXTERNAL_UNSAT;
    } else if (optimum && modelFound) {
        result.status = EXTERNAL_OPTIMUM;
    } else if (satisfiable && modelFound) {
        result.status = EXTERNAL_TIMEOUT_FEASIBLE;
        result.message = "solver returned a feasible incumbent without an optimality proof";
    } else {
        result.status = EXTERNAL_ERROR;
        result.message = "solver did not return a recognized status with a model";
    }
    return result;
}

static ExternalResult runExternalSolver(
    const string &solverPath,
    const string &formulaPath,
    const string &outputPath,
    int variables,
    double timeoutSeconds,
    double shutdownGraceSeconds,
    bool alignEvalMaxSATTargetTime
) {
    ExternalResult result;
    if (timeoutSeconds <= 0) {
        result.status = EXTERNAL_TIMEOUT;
        return result;
    }

    auto started = chrono::steady_clock::now();
    pid_t child = fork();
    if (child < 0)
        throw runtime_error("fork failed");
    if (child == 0) {
        int output = open(outputPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (output < 0)
            _exit(126);
        dup2(output, STDOUT_FILENO);
        dup2(output, STDERR_FILENO);
        close(output);
        if (alignEvalMaxSATTargetTime) {
            const double boundedTarget = max(1.0, floor(timeoutSeconds));
            if (boundedTarget > numeric_limits<unsigned int>::max())
                _exit(125);
            const string target = to_string(
                static_cast<unsigned int>(boundedTarget)
            );
            execlp(
                solverPath.c_str(),
                solverPath.c_str(),
                "--TCT",
                target.c_str(),
                formulaPath.c_str(),
                static_cast<char *>(NULL)
            );
        } else {
            execlp(
                solverPath.c_str(),
                solverPath.c_str(),
                formulaPath.c_str(),
                static_cast<char *>(NULL)
            );
        }
        _exit(127);
    }

    int childStatus = 0;
    result.solverLaunched = true;
    bool timedOut = false;
    double solverElapsed = 0.0;
    double shutdownElapsed = 0.0;
    while (true) {
        pid_t checked = waitpid(child, &childStatus, WNOHANG);
        if (checked == child)
            break;
        if (checked < 0)
            throw runtime_error("waitpid failed");
        double elapsed = chrono::duration<double>(
            chrono::steady_clock::now() - started
        ).count();
        if (elapsed >= timeoutSeconds) {
            timedOut = true;
            solverElapsed = timeoutSeconds;
            kill(child, SIGTERM);
            const auto shutdownStarted = chrono::steady_clock::now();
            const auto shutdownDeadline = shutdownStarted +
                chrono::duration_cast<chrono::steady_clock::duration>(
                    chrono::duration<double>(shutdownGraceSeconds)
                );
            while (waitpid(child, &childStatus, WNOHANG) == 0 &&
                   chrono::steady_clock::now() < shutdownDeadline) {
                this_thread::sleep_for(chrono::milliseconds(10));
            }
            if (waitpid(child, &childStatus, WNOHANG) == 0) {
                kill(child, SIGKILL);
                waitpid(child, &childStatus, 0);
            }
            shutdownElapsed = chrono::duration<double>(
                chrono::steady_clock::now() - shutdownStarted
            ).count();
            break;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    const double elapsed = chrono::duration<double>(
        chrono::steady_clock::now() - started
    ).count();
    if (timedOut) {
        ExternalResult parsed = parseSolverOutput(
            outputPath, variables, solverElapsed
        );
        parsed.solverLaunched = true;
        parsed.shutdownGraceSeconds = shutdownElapsed;
        if (parsed.status == EXTERNAL_OPTIMUM ||
            parsed.status == EXTERNAL_UNSAT ||
            parsed.status == EXTERNAL_TIMEOUT_FEASIBLE)
            return parsed;
        result.status = EXTERNAL_TIMEOUT;
        result.elapsedSeconds = solverElapsed;
        result.shutdownGraceSeconds = shutdownElapsed;
        result.message = "solver reached the external time limit without a usable model";
        return result;
    }
    ExternalResult parsed = parseSolverOutput(outputPath, variables, elapsed);
    parsed.solverLaunched = true;
    if (parsed.status == EXTERNAL_ERROR) {
        if (WIFSIGNALED(childStatus)) {
            parsed.message = "solver terminated by signal "
                + to_string(WTERMSIG(childStatus));
        } else if (WIFEXITED(childStatus) && WEXITSTATUS(childStatus) != 0) {
            parsed.message = "solver exited with code "
                + to_string(WEXITSTATUS(childStatus));
        }
    }
    return parsed;
}

static void writeLegacyWCNF(ostream &output, SMTFormula *formula) {
    const int top = formula->getHardWeight();
    output << "p wcnf " << formula->getNBoolVars() << ' '
           << formula->getNClauses() + formula->getNSoftClauses() << ' '
           << top << '\n';

    for (const clause &hardClause : formula->getClauses()) {
        output << top << ' ';
        for (const literal &value : hardClause.v)
            output << (value.sign ? value.v.id : -value.v.id) << ' ';
        output << "0\n";
    }
    for (int index = 0; index < formula->getNSoftClauses(); ++index) {
        output << formula->getWeights()[index] << ' ';
        for (const literal &value : formula->getSoftClauses()[index].v)
            output << (value.sign ? value.v.id : -value.v.id) << ' ';
        output << "0\n";
    }
}

static string objectiveName(HCORAPObjectiveKind objective) {
    switch (objective) {
        case HCORAP_COVERAGE: return "coverage";
        case HCORAP_SIMILARITY: return "similarity";
        case HCORAP_CONTINUITY: return "continuity";
        case HCORAP_OVERTIME: return "overtime";
        case HCORAP_LEX_COS_SINGLE:
        case HCORAP_LEX_OCS_SINGLE:
            return "lexicographic_score";
        case HCORAP_WEIGHTED:
        default: return "weighted_score";
    }
}

static string objectiveSense(HCORAPObjectiveKind objective) {
    return objective == HCORAP_CONTINUITY || objective == HCORAP_OVERTIME
        ? "min" : "max";
}

static string objectiveMode(const string &method) {
    if (method == "weighted")
        return "weighted";
    if (method == "lex-continuity" || method == "lex-cos" ||
        method == "lex-overtime" || method == "lex-cos-one-shot" ||
        method == "lex-overtime-one-shot")
        return "lexicographic";
    return "epsilon-constraint";
}

static string objectivePolicy(const string &method) {
    if (method == "lex-continuity")
        return "continuity-priority";
    if (method == "lex-cos")
        return "continuity-overtime-similarity";
    if (method == "lex-cos-one-shot")
        return "continuity-overtime-similarity";
    if (method == "lex-overtime")
        return "overtime-priority";
    if (method == "lex-overtime-one-shot")
        return "overtime-priority";
    if (method == "epsilon")
        return "similarity-budget";
    return "weighted-sum";
}

static string lexicographicImplementation(const string &method) {
    if (method == "lex-cos-one-shot" || method == "lex-overtime-one-shot")
        return "single-call-dominance-weights";
    if (method == "lex-continuity" || method == "lex-cos" ||
        method == "lex-overtime")
        return "sequential-stages";
    return "not-applicable";
}

static string temporaryBase(int stageIndex) {
    ostringstream name;
    name << "/tmp/hcorap_multi_" << static_cast<long>(getpid())
         << "_" << stageIndex;
    return name.str();
}

static ExternalStatus solveStage(
    HCORAP *instance,
    const Options &options,
    HCORAPObjectiveKind objective,
    const HCORAPObjectiveBounds &bounds,
    int stageIndex,
    const chrono::steady_clock::time_point &overallStarted,
    StageRecord &record,
    HCORAPSolutionMetrics &metrics,
    string &error
) {
    auto encodeStarted = chrono::steady_clock::now();
    HCORAPMultiObjectiveEncoding encoding(
        instance,
        objective,
        options.fullCoverage,
        options.continuityWeight,
        options.overtimeWeight,
        options.cardinalityEncoding,
        options.impliedConfig,
        options.symmetryBreaking,
        bounds
    );
    SMTFormula *formula = encoding.encode();
    double encodeSeconds = chrono::duration<double>(
        chrono::steady_clock::now() - encodeStarted
    ).count();

    string base = temporaryBase(stageIndex);
    string formulaPath = base + ".wcnf";
    string solverOutputPath = base + ".out";
    ofstream formulaOutput(formulaPath.c_str());
    if (!formulaOutput) {
        delete formula;
        throw runtime_error("cannot create temporary WCNF: " + formulaPath);
    }
    writeLegacyWCNF(formulaOutput, formula);
    formulaOutput.close();

    double totalElapsed = chrono::duration<double>(
        chrono::steady_clock::now() - overallStarted
    ).count();
    double remaining = options.timeoutSeconds - totalElapsed;
    ExternalResult solved = runExternalSolver(
        options.solverPath,
        formulaPath,
        solverOutputPath,
        formula->getNBoolVars(),
        remaining,
        options.solverShutdownGraceSeconds,
        options.alignEvalMaxSATTargetTime
    );

    record.name = objectiveName(objective);
    record.sense = objectiveSense(objective);
    record.encodeSeconds = encodeSeconds;
    record.solveSeconds = solved.elapsedSeconds;
    record.solverTargetSeconds = options.alignEvalMaxSATTargetTime
        && solved.solverLaunched ? max(1.0, floor(remaining))
        : 0.0;
    record.hasSolverTarget =
        options.alignEvalMaxSATTargetTime && solved.solverLaunched;
    record.solverLaunched = solved.solverLaunched;
    record.shutdownGraceSeconds = solved.shutdownGraceSeconds;
    record.variables = formula->getNBoolVars();
    record.hardClauses = formula->getNClauses();
    record.softClauses = formula->getNSoftClauses();
    record.continuityObjectiveWeight =
        encoding.effectiveContinuityObjectiveWeight();
    record.overtimeObjectiveWeight =
        encoding.effectiveOvertimeObjectiveWeight();

    if (solved.status == EXTERNAL_OPTIMUM ||
        solved.status == EXTERNAL_TIMEOUT_FEASIBLE) {
        encoding.setBooleanModel(solved.model);
        metrics = encoding.evaluateModel();
        bool boundsSatisfied =
            (bounds.minCoverage < 0 || metrics.coverage >= bounds.minCoverage) &&
            (bounds.minSimilarity < 0 || metrics.similarity >= bounds.minSimilarity) &&
            (bounds.maxContinuity < 0 || metrics.continuity <= bounds.maxContinuity) &&
            (bounds.maxOvertime < 0 || metrics.overtime <= bounds.maxOvertime);
        if (!metrics.valid || !boundsSatisfied) {
            solved.status = EXTERNAL_ERROR;
            error = !metrics.valid
                ? "independent C++ verifier rejected solver model"
                : "independent C++ verifier rejected an objective bound";
            solved.message = error;
        } else {
            record.incumbent = encoding.objectiveValue(metrics);
            record.hasIncumbent = true;
            if (solved.status == EXTERNAL_OPTIMUM) {
                record.optimum = record.incumbent;
                record.hasOptimum = true;
            }
        }
    } else if (solved.status == EXTERNAL_ERROR && !solved.message.empty()) {
        error = solved.message;
    }
    record.status = statusName(solved.status);
    record.message = solved.message;

    delete formula;
    if (!options.keepFiles) {
        remove(formulaPath.c_str());
        remove(solverOutputPath.c_str());
    }
    return solved.status;
}

static void updateBound(
    HCORAPObjectiveBounds &bounds,
    HCORAPObjectiveKind objective,
    const HCORAPSolutionMetrics &metrics
) {
    switch (objective) {
        case HCORAP_COVERAGE:
            bounds.minCoverage = metrics.coverage;
            break;
        case HCORAP_SIMILARITY:
            bounds.minSimilarity = metrics.similarity;
            break;
        case HCORAP_CONTINUITY:
            bounds.maxContinuity = metrics.continuity;
            break;
        case HCORAP_OVERTIME:
            bounds.maxOvertime = metrics.overtime;
            break;
        default:
            break;
    }
}

static ExternalStatus executeAndRecord(
    HCORAP *instance,
    const Options &options,
    HCORAPObjectiveKind objective,
    HCORAPObjectiveBounds &bounds,
    const chrono::steady_clock::time_point &overallStarted,
    RunState &state,
    HCORAPSolutionMetrics &metrics,
    bool preserveOptimum
) {
    StageRecord record;
    ExternalStatus status = solveStage(
        instance,
        options,
        objective,
        bounds,
        static_cast<int>(state.stages.size()),
        overallStarted,
        record,
        metrics,
        state.error
    );
    if (record.solverLaunched)
        ++state.solverCalls;
    state.stages.push_back(record);
    if (record.hasIncumbent) {
        state.metrics = metrics;
        state.hasIncumbent = true;
        state.incumbentStageIndex = static_cast<int>(state.stages.size()) - 1;
    }
    if (status == EXTERNAL_OPTIMUM) {
        ++state.provedStages;
        if (preserveOptimum)
            updateBound(bounds, objective, metrics);
    }
    return status;
}

static vector<HCORAPObjectiveKind> lexicographicOrder(const string &method) {
    if (method == "lex-continuity") {
        return {
            HCORAP_CONTINUITY,
            HCORAP_SIMILARITY,
            HCORAP_OVERTIME
        };
    }
    if (method == "lex-cos") {
        return {
            HCORAP_CONTINUITY,
            HCORAP_OVERTIME,
            HCORAP_SIMILARITY
        };
    }
    if (method == "lex-overtime") {
        return {
            HCORAP_OVERTIME,
            HCORAP_CONTINUITY,
            HCORAP_SIMILARITY
        };
    }
    throw runtime_error("not a lexicographic method: " + method);
}

static ExternalStatus executeLexicographicPolicy(
    HCORAP *instance,
    const Options &options,
    HCORAPObjectiveBounds &bounds,
    const chrono::steady_clock::time_point &overallStarted,
    RunState &state,
    HCORAPSolutionMetrics &metrics
) {
    ExternalStatus status = EXTERNAL_OPTIMUM;
    const vector<HCORAPObjectiveKind> order = lexicographicOrder(options.method);
    for (size_t index = 0; index < order.size(); ++index) {
        const HCORAPObjectiveKind objective = order[index];
        if (index == 2 && objective == HCORAP_SIMILARITY &&
            options.stage3IncumbentBound) {
            bounds.minSimilarity = metrics.similarity;
            state.similarityLowerBound = metrics.similarity;
        }
        status = executeAndRecord(
            instance,
            options,
            objective,
            bounds,
            overallStarted,
            state,
            metrics,
            true
        );
        if (status != EXTERNAL_OPTIMUM)
            break;
    }
    return status;
}

static const StageRecord *findStage(
    const RunState &state,
    const string &name
) {
    for (const StageRecord &stage : state.stages) {
        if (stage.name == name)
            return &stage;
    }
    return NULL;
}

static bool validateEpsilonCompletion(
    const Options &options,
    const RunState &state,
    const HCORAPSolutionMetrics &metrics,
    string &error
) {
    const StageRecord *reference = findStage(state, "similarity_reference");
    const StageRecord *continuity = findStage(state, "continuity");
    const StageRecord *overtime = findStage(state, "overtime");
    const StageRecord *tiebreak = findStage(state, "similarity_tiebreak");
    if (reference == NULL || continuity == NULL || overtime == NULL ||
        tiebreak == NULL) {
        error = "epsilon completion is missing one or more optimization stages";
        return false;
    }

    int expectedLowerBound = similarityThreshold(
        state.similarityReferenceOptimum, options.delta
    );
    if (
        reference->optimum != state.similarityReferenceOptimum ||
        state.similarityLowerBound != expectedLowerBound
    ) {
        error = "epsilon similarity reference or lower bound is inconsistent";
        return false;
    }
    if (
        metrics.similarity < state.similarityLowerBound ||
        metrics.similarity > state.similarityReferenceOptimum
    ) {
        error = "epsilon final similarity violates its certified budget interval";
        return false;
    }
    if (
        continuity->optimum != metrics.continuity ||
        overtime->optimum != metrics.overtime ||
        tiebreak->optimum != metrics.similarity
    ) {
        error = "epsilon final metrics do not match the sequential stage optima";
        return false;
    }

    if (!options.fullCoverage) {
        const StageRecord *coverage = findStage(state, "coverage");
        if (coverage == NULL || coverage->optimum != metrics.coverage) {
            error = "epsilon final coverage does not match the coverage optimum";
            return false;
        }
    }
    return true;
}

static ExternalStatus executeEpsilonPolicy(
    HCORAP *instance,
    const Options &options,
    HCORAPObjectiveBounds &bounds,
    const chrono::steady_clock::time_point &overallStarted,
    RunState &state,
    HCORAPSolutionMetrics &metrics
) {
    HCORAPSolutionMetrics referenceMetrics;
    ExternalStatus status = executeAndRecord(
        instance,
        options,
        HCORAP_SIMILARITY,
        bounds,
        overallStarted,
        state,
        referenceMetrics,
        false
    );
    if (status != EXTERNAL_OPTIMUM)
        return status;

    state.stages.back().name = "similarity_reference";
    state.similarityReferenceOptimum = referenceMetrics.similarity;
    state.similarityLowerBound = similarityThreshold(
        state.similarityReferenceOptimum, options.delta
    );
    bounds.minSimilarity = state.similarityLowerBound;

    const HCORAPObjectiveKind completion[] = {
        HCORAP_CONTINUITY,
        HCORAP_OVERTIME,
        HCORAP_SIMILARITY
    };
    for (size_t index = 0; index < 3; ++index) {
        status = executeAndRecord(
            instance,
            options,
            completion[index],
            bounds,
            overallStarted,
            state,
            metrics,
            index + 1 < 3
        );
        if (status != EXTERNAL_OPTIMUM)
            return status;
    }

    state.stages.back().name = "similarity_tiebreak";
    if (!validateEpsilonCompletion(options, state, metrics, state.error))
        return EXTERNAL_ERROR;
    return EXTERNAL_OPTIMUM;
}

static string statusName(ExternalStatus status) {
    switch (status) {
        case EXTERNAL_OPTIMUM: return "OPTIMUM";
        case EXTERNAL_UNSAT: return "UNSATISFIABLE";
        case EXTERNAL_TIMEOUT: return "TIMEOUT";
        case EXTERNAL_TIMEOUT_FEASIBLE: return "TIMEOUT_FEASIBLE";
        case EXTERNAL_ERROR:
        default: return "ERROR";
    }
}

static RunState solveMethod(
    HCORAP *instance,
    const Options &options,
    const chrono::steady_clock::time_point &overallStarted
) {
    RunState state;
    HCORAPObjectiveBounds bounds;
    HCORAPSolutionMetrics metrics;
    ExternalStatus status = EXTERNAL_OPTIMUM;

    if (!options.fullCoverage) {
        status = executeAndRecord(
            instance, options, HCORAP_COVERAGE, bounds, overallStarted,
            state, metrics, true
        );
    }

    if (status == EXTERNAL_OPTIMUM && options.method == "weighted") {
        status = executeAndRecord(
            instance, options, HCORAP_WEIGHTED, bounds, overallStarted,
            state, metrics, false
        );
    } else if (
        status == EXTERNAL_OPTIMUM &&
        (options.method == "lex-continuity" ||
         options.method == "lex-cos" ||
         options.method == "lex-overtime")
    ) {
        status = executeLexicographicPolicy(
            instance, options, bounds, overallStarted, state, metrics
        );
    } else if (
        status == EXTERNAL_OPTIMUM &&
        (options.method == "lex-cos-one-shot" ||
         options.method == "lex-overtime-one-shot")
    ) {
        const HCORAPObjectiveKind objective =
            options.method == "lex-cos-one-shot"
            ? HCORAP_LEX_COS_SINGLE
            : HCORAP_LEX_OCS_SINGLE;
        status = executeAndRecord(
            instance, options, objective, bounds, overallStarted,
            state, metrics, false
        );
    } else if (status == EXTERNAL_OPTIMUM && options.method == "epsilon") {
        status = executeEpsilonPolicy(
            instance, options, bounds, overallStarted, state, metrics
        );
    }

    state.status = status == EXTERNAL_TIMEOUT && state.hasIncumbent
        ? "TIMEOUT_FEASIBLE"
        : statusName(status);
    if (status == EXTERNAL_OPTIMUM || state.hasIncumbent)
        state.metrics = metrics;
    return state;
}

static int certifiedLexicographicPrefix(
    const Options &options, const RunState &state
) {
    if (options.method == "lex-cos-one-shot" ||
        options.method == "lex-overtime-one-shot")
        return state.status == "OPTIMUM" ? 3 : 0;
    if (options.method != "lex-continuity" && options.method != "lex-cos" &&
        options.method != "lex-overtime")
        return -1;

    const vector<HCORAPObjectiveKind> order = lexicographicOrder(options.method);
    size_t expected = 0;
    for (const StageRecord &stage : state.stages) {
        if (expected >= order.size())
            break;
        if (stage.name == objectiveName(order[expected]) &&
            stage.status == "OPTIMUM") {
            ++expected;
        } else if (stage.name != "coverage") {
            break;
        }
    }
    return static_cast<int>(expected);
}

static void jsonEscape(ostream &output, const string &value) {
    output << '"';
    for (char character : value) {
        if (character == '"' || character == '\\')
            output << '\\';
        if (character == '\n')
            output << "\\n";
        else
            output << character;
    }
    output << '"';
}

static void writeResult(
    ostream &output,
    const RunState &state,
    const Options &options,
    double totalSeconds,
    const HCORAP *instance
) {
    output << "{\n  \"schema_version\": 3,\n  \"status\": ";
    jsonEscape(output, state.status);
    output << ",\n  \"method\": ";
    jsonEscape(output, options.method);
    output << ",\n  \"objective_mode\": ";
    jsonEscape(output, objectiveMode(options.method));
    output << ",\n  \"objective_policy\": ";
    jsonEscape(output, objectivePolicy(options.method));
    output << ",\n  \"lexicographic_implementation\": ";
    jsonEscape(output, lexicographicImplementation(options.method));
    output << ",\n  \"language\": \"C++\",\n  \"instance\": ";
    jsonEscape(output, options.instancePath);
    output << ",\n  \"solver\": ";
    jsonEscape(output, options.solverPath);
    output << ",\n  \"wcnf_format\": \"legacy-top-weight\",\n"
           << "  \"timing_scope\": \"parse+encode+serialize+solve+verify\",\n"
           << "  \"timeout_seconds\": " << options.timeoutSeconds << ",\n"
           << "  \"solver_shutdown_grace_seconds\": "
           << options.solverShutdownGraceSeconds << ",\n"
           << "  \"align_evalmaxsat_tct\": "
           << (options.alignEvalMaxSATTargetTime ? "true" : "false") << ",\n"
           << "  \"stage3_incumbent_bound\": "
           << (options.stage3IncumbentBound ? "true" : "false") << ",\n"
           << "  \"elapsed_seconds\": " << setprecision(10) << totalSeconds << ",\n"
           << "  \"solver_calls\": " << state.solverCalls << ",\n"
           << "  \"proved_stage_count\": " << state.provedStages << ",\n"
           << "  \"certified_lexicographic_prefix\": ";
    const int certifiedPrefix = certifiedLexicographicPrefix(options, state);
    if (certifiedPrefix >= 0)
        output << certifiedPrefix;
    else
        output << "null";
    output << ",\n"
           << "  \"incumbent_stage_index\": ";
    if (state.incumbentStageIndex >= 0)
        output << state.incumbentStageIndex;
    else
        output << "null";
    output << ",\n"
           << "  \"full_coverage\": " << (options.fullCoverage ? "true" : "false") << ",\n"
           << "  \"continuity_weight\": " << options.continuityWeight << ",\n"
           << "  \"overtime_weight\": " << options.overtimeWeight << ",\n"
           << "  \"overtime_penalty_per_hour\": " << abs(instance->P) << ",\n"
           << "  \"cardinality_encoding\": \""
           << hcorapCardinalityEncodingName(options.cardinalityEncoding)
           << "\",\n"
           << "  \"implied_constraints\": \""
           << hcorapImpliedConfigName(options.impliedConfig)
           << "\",\n"
           << "  \"symmetry_breaking\": \""
           << hcorapSymmetryBreakingName(options.symmetryBreaking)
           << "\",\n"
           << "  \"delta\": ";
    jsonEscape(output, options.delta);
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
    output << ",\n  \"similarity_realized_loss_absolute\": ";
    if (state.status == "OPTIMUM" && state.similarityReferenceOptimum >= 0) {
        output << state.similarityReferenceOptimum - state.metrics.similarity;
    } else {
        output << "null";
    }
    output << ",\n  \"similarity_realized_loss_fraction\": ";
    if (state.status == "OPTIMUM" && state.similarityReferenceOptimum >= 0) {
        int loss = state.similarityReferenceOptimum - state.metrics.similarity;
        ostringstream fraction;
        if (state.similarityReferenceOptimum == 0 || loss == 0) {
            fraction << "0/1";
        } else {
            int divisor = greatestCommonDivisor(
                loss, state.similarityReferenceOptimum
            );
            fraction
                << loss / divisor
                << '/'
                << state.similarityReferenceOptimum / divisor;
        }
        jsonEscape(output, fraction.str());
    } else {
        output << "null";
    }
    output << ",\n  \"stages\": [";
    for (size_t index = 0; index < state.stages.size(); ++index) {
        const StageRecord &stage = state.stages[index];
        if (index)
            output << ',';
        output << "\n    {\"objective\": ";
        jsonEscape(output, stage.name);
        output << ", \"sense\": ";
        jsonEscape(output, stage.sense);
        output << ", \"status\": ";
        jsonEscape(output, stage.status);
        output << ", \"optimum\": ";
        if (stage.hasOptimum)
            output << stage.optimum;
        else
            output << "null";
        output << ", \"incumbent\": ";
        if (stage.hasIncumbent)
            output << stage.incumbent;
        else
            output << "null";
        output << ", \"encode_seconds\": " << stage.encodeSeconds
               << ", \"solve_seconds\": " << stage.solveSeconds
               << ", \"solver_target_seconds\": ";
        if (stage.hasSolverTarget)
            output << stage.solverTargetSeconds;
        else
            output << "null";
        output << ", \"shutdown_grace_seconds\": "
               << stage.shutdownGraceSeconds
               << ", \"solver_launched\": "
               << (stage.solverLaunched ? "true" : "false")
               << ", \"variables\": " << stage.variables
               << ", \"hard_clauses\": " << stage.hardClauses
               << ", \"soft_clauses\": " << stage.softClauses
               << ", \"continuity_objective_weight\": "
               << stage.continuityObjectiveWeight
               << ", \"overtime_objective_weight\": "
               << stage.overtimeObjectiveWeight;
        if (!stage.message.empty()) {
            output << ", \"message\": ";
            jsonEscape(output, stage.message);
        }
        output << '}';
    }
    if (!state.stages.empty())
        output << '\n';
    output << "  ],\n";
    if (!state.error.empty()) {
        output << "  \"error\": ";
        jsonEscape(output, state.error);
        output << ",\n";
    }
    if (state.status == "OPTIMUM" || state.hasIncumbent) {
        int weightedReferenceScore =
            state.metrics.similarity
            - options.continuityWeight * state.metrics.continuity
            - options.overtimeWeight * abs(instance->P) * state.metrics.overtime;
        output << "  \"metrics\": {\"coverage\": " << state.metrics.coverage
               << ", \"similarity\": " << state.metrics.similarity
               << ", \"continuity\": " << state.metrics.continuity
               << ", \"overtime\": " << state.metrics.overtime
               << ", \"overtime_cost\": " << state.metrics.overtimeCost
               << ", \"weighted_reference_score\": " << weightedReferenceScore
               << ", \"verified\": " << (state.metrics.valid ? "true" : "false")
               << "}";
        if (options.printAssignments) {
            output << ",\n  \"assignments\": [";
            for (size_t index = 0; index < state.metrics.assignments.size(); ++index) {
                if (index)
                    output << ',';
                output << "[" << get<0>(state.metrics.assignments[index])
                       << ',' << get<1>(state.metrics.assignments[index])
                       << ',' << get<2>(state.metrics.assignments[index]) << "]";
            }
            output << ']';
        }
        output << '\n';
    } else {
        output << "  \"metrics\": null\n";
    }
    output << "}\n";
}

int main(int argc, char **argv) {
    try {
        Options options = parseOptions(argc, argv);
        auto started = chrono::steady_clock::now();
        HCORAP *instance = parser::parseHCORAP(options.instancePath);

        if (options.encodeOnly) {
            HCORAPObjectiveBounds bounds;
            HCORAPMultiObjectiveEncoding encoding(
                instance,
                HCORAP_WEIGHTED,
                true,
                options.continuityWeight,
                options.overtimeWeight,
                options.cardinalityEncoding,
                options.impliedConfig,
                options.symmetryBreaking,
                bounds
            );
            SMTFormula *formula = encoding.encode();
            writeLegacyWCNF(cout, formula);
            delete formula;
            delete instance;
            return 0;
        }

        RunState state = solveMethod(instance, options, started);
        double totalSeconds = chrono::duration<double>(
            chrono::steady_clock::now() - started
        ).count();

        if (options.outputPath.empty()) {
            writeResult(cout, state, options, totalSeconds, instance);
        } else {
            ofstream output(options.outputPath.c_str());
            if (!output)
                throw runtime_error("cannot open output file: " + options.outputPath);
            writeResult(output, state, options, totalSeconds, instance);
        }
        delete instance;
        return state.status == "OPTIMUM" ? 0 : 2;
    } catch (const exception &error) {
        cerr << "ERROR: " << error.what() << endl;
        usage(argv[0]);
        return 2;
    }
}
