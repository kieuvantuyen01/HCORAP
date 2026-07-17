#include "parser.h"
#include "HCORAPMultiObjectiveEncoding.h"
#include "dimacsfileencoder.h"

#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    int continuityWeight;
    int overtimeWeight;
    HCORAPCardinalityEncoding cardinalityEncoding;
    HCORAPImpliedConfig impliedConfig;
    HCORAPSymmetryBreaking symmetryBreaking;
    bool fullCoverage;
    bool encodeOnly;
    bool keepFiles;
    bool printAssignments;

    Options()
        : method("weighted"), solverPath("open-wbo"), delta("0.05"),
          timeoutSeconds(3600.0), continuityWeight(1), overtimeWeight(1),
          cardinalityEncoding(HCORAP_SORTING_NETWORK),
          impliedConfig(HCORAP_IMPLIED_NONE),
          symmetryBreaking(HCORAP_SYMMETRY_NONE),
          fullCoverage(true), encodeOnly(false), keepFiles(false),
          printAssignments(false) {}
};

enum ExternalStatus {
    EXTERNAL_OPTIMUM,
    EXTERNAL_UNSAT,
    EXTERNAL_TIMEOUT,
    EXTERNAL_ERROR
};

struct ExternalResult {
    ExternalStatus status;
    vector<bool> model;
    double elapsedSeconds;
    string message;

    ExternalResult() : status(EXTERNAL_ERROR), elapsedSeconds(0.0) {}
};

struct StageRecord {
    string name;
    string sense;
    int optimum;
    double encodeSeconds;
    double solveSeconds;
    int variables;
    int hardClauses;
    int softClauses;
};

struct RunState {
    string status;
    vector<StageRecord> stages;
    HCORAPSolutionMetrics metrics;
    string error;
};

static void usage(const char *program) {
    cerr
        << "Usage: " << program << " INSTANCE [options]\n"
        << "  --method weighted|lex-continuity|lex-overtime|epsilon\n"
        << "  --solver PATH             Open-WBO-compatible C++ solver\n"
        << "  --timeout SECONDS         cumulative encode+solve timeout\n"
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
    if (options.timeoutSeconds <= 0)
        throw runtime_error("timeout must be positive");
    if (options.continuityWeight < 0 || options.overtimeWeight < 0)
        throw runtime_error("wc and wo must be non-negative");
    if (options.method != "weighted" && options.method != "lex-continuity" &&
        options.method != "lex-overtime" && options.method != "epsilon")
        throw runtime_error("unsupported method: " + options.method);
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
    } else {
        result.status = EXTERNAL_ERROR;
        result.message = "solver did not return an optimum with a model";
    }
    return result;
}

static ExternalResult runExternalSolver(
    const string &solverPath,
    const string &formulaPath,
    const string &outputPath,
    int variables,
    double timeoutSeconds
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
        execlp(
            solverPath.c_str(),
            solverPath.c_str(),
            formulaPath.c_str(),
            static_cast<char *>(NULL)
        );
        _exit(127);
    }

    int childStatus = 0;
    bool timedOut = false;
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
            kill(child, SIGTERM);
            this_thread::sleep_for(chrono::milliseconds(100));
            if (waitpid(child, &childStatus, WNOHANG) == 0) {
                kill(child, SIGKILL);
                waitpid(child, &childStatus, 0);
            }
            break;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
    double elapsed = chrono::duration<double>(
        chrono::steady_clock::now() - started
    ).count();
    if (timedOut) {
        result.status = EXTERNAL_TIMEOUT;
        result.elapsedSeconds = elapsed;
        return result;
    }
    ExternalResult parsed = parseSolverOutput(outputPath, variables, elapsed);
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
        case HCORAP_WEIGHTED:
        default: return "weighted_score";
    }
}

static string objectiveSense(HCORAPObjectiveKind objective) {
    return objective == HCORAP_CONTINUITY || objective == HCORAP_OVERTIME
        ? "min" : "max";
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
        remaining
    );

    record.name = objectiveName(objective);
    record.sense = objectiveSense(objective);
    record.encodeSeconds = encodeSeconds;
    record.solveSeconds = solved.elapsedSeconds;
    record.variables = formula->getNBoolVars();
    record.hardClauses = formula->getNClauses();
    record.softClauses = formula->getNSoftClauses();

    if (solved.status == EXTERNAL_OPTIMUM) {
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
        } else {
            record.optimum = encoding.objectiveValue(metrics);
        }
    } else if (!solved.message.empty()) {
        error = solved.message;
    }

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
    if (status == EXTERNAL_OPTIMUM) {
        state.stages.push_back(record);
        if (preserveOptimum)
            updateBound(bounds, objective, metrics);
    }
    return status;
}

static string statusName(ExternalStatus status) {
    switch (status) {
        case EXTERNAL_OPTIMUM: return "OPTIMUM";
        case EXTERNAL_UNSAT: return "UNSATISFIABLE";
        case EXTERNAL_TIMEOUT: return "TIMEOUT";
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
    } else if (status == EXTERNAL_OPTIMUM && options.method == "lex-continuity") {
        const HCORAPObjectiveKind order[] = {
            HCORAP_CONTINUITY, HCORAP_SIMILARITY, HCORAP_OVERTIME
        };
        for (HCORAPObjectiveKind objective : order) {
            status = executeAndRecord(
                instance, options, objective, bounds, overallStarted,
                state, metrics, true
            );
            if (status != EXTERNAL_OPTIMUM)
                break;
        }
    } else if (status == EXTERNAL_OPTIMUM && options.method == "lex-overtime") {
        const HCORAPObjectiveKind order[] = {
            HCORAP_OVERTIME, HCORAP_CONTINUITY, HCORAP_SIMILARITY
        };
        for (HCORAPObjectiveKind objective : order) {
            status = executeAndRecord(
                instance, options, objective, bounds, overallStarted,
                state, metrics, true
            );
            if (status != EXTERNAL_OPTIMUM)
                break;
        }
    } else if (status == EXTERNAL_OPTIMUM && options.method == "epsilon") {
        HCORAPSolutionMetrics reference;
        status = executeAndRecord(
            instance, options, HCORAP_SIMILARITY, bounds, overallStarted,
            state, reference, false
        );
        if (status == EXTERNAL_OPTIMUM) {
            state.stages.back().name = "similarity_reference";
            bounds.minSimilarity = similarityThreshold(
                reference.similarity, options.delta
            );
            const HCORAPObjectiveKind order[] = {
                HCORAP_CONTINUITY, HCORAP_OVERTIME, HCORAP_SIMILARITY
            };
            for (HCORAPObjectiveKind objective : order) {
                status = executeAndRecord(
                    instance, options, objective, bounds, overallStarted,
                    state, metrics, objective != HCORAP_SIMILARITY
                );
                if (status != EXTERNAL_OPTIMUM)
                    break;
            }
            if (status == EXTERNAL_OPTIMUM)
                state.stages.back().name = "similarity_tiebreak";
        }
    }

    state.status = statusName(status);
    if (status == EXTERNAL_OPTIMUM)
        state.metrics = metrics;
    return state;
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
    double totalSeconds
) {
    output << "{\n  \"schema_version\": 1,\n  \"status\": ";
    jsonEscape(output, state.status);
    output << ",\n  \"method\": ";
    jsonEscape(output, options.method);
    output << ",\n  \"language\": \"C++\",\n  \"instance\": ";
    jsonEscape(output, options.instancePath);
    output << ",\n  \"solver\": ";
    jsonEscape(output, options.solverPath);
    output << ",\n  \"wcnf_format\": \"legacy-top-weight\",\n"
           << "  \"timing_scope\": \"parse+encode+serialize+solve+verify\",\n"
           << "  \"timeout_seconds\": " << options.timeoutSeconds << ",\n"
           << "  \"elapsed_seconds\": " << setprecision(10) << totalSeconds << ",\n"
           << "  \"full_coverage\": " << (options.fullCoverage ? "true" : "false") << ",\n"
           << "  \"continuity_weight\": " << options.continuityWeight << ",\n"
           << "  \"overtime_weight\": " << options.overtimeWeight << ",\n"
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
    output << ",\n  \"stages\": [";
    for (size_t index = 0; index < state.stages.size(); ++index) {
        const StageRecord &stage = state.stages[index];
        if (index)
            output << ',';
        output << "\n    {\"objective\": ";
        jsonEscape(output, stage.name);
        output << ", \"sense\": ";
        jsonEscape(output, stage.sense);
        output << ", \"optimum\": " << stage.optimum
               << ", \"encode_seconds\": " << stage.encodeSeconds
               << ", \"solve_seconds\": " << stage.solveSeconds
               << ", \"variables\": " << stage.variables
               << ", \"hard_clauses\": " << stage.hardClauses
               << ", \"soft_clauses\": " << stage.softClauses << '}';
    }
    if (!state.stages.empty())
        output << '\n';
    output << "  ],\n";
    if (!state.error.empty()) {
        output << "  \"error\": ";
        jsonEscape(output, state.error);
        output << ",\n";
    }
    if (state.status == "OPTIMUM") {
        output << "  \"metrics\": {\"coverage\": " << state.metrics.coverage
               << ", \"similarity\": " << state.metrics.similarity
               << ", \"continuity\": " << state.metrics.continuity
               << ", \"overtime\": " << state.metrics.overtime
               << ", \"overtime_cost\": " << state.metrics.overtimeCost
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
            writeResult(cout, state, options, totalSeconds);
        } else {
            ofstream output(options.outputPath.c_str());
            if (!output)
                throw runtime_error("cannot open output file: " + options.outputPath);
            writeResult(output, state, options, totalSeconds);
        }
        delete instance;
        return state.status == "OPTIMUM" ? 0 : 2;
    } catch (const exception &error) {
        cerr << "ERROR: " << error.what() << endl;
        usage(argv[0]);
        return 2;
    }
}
