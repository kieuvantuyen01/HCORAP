#include "CardinalityNetwork.h"

#include <stdexcept>

using namespace std;

namespace {

// Build a bidirectional totalizer.  outputs[k] is true iff at least k + 1
// inputs are true, matching the semantics exposed by SMTFormula::addSorting.
void addExactTotalizer(
    SMTFormula *formula,
    const vector<literal> &inputs,
    vector<literal> &outputs
) {
    const int size = static_cast<int>(inputs.size());
    if (size == 0) {
        outputs.clear();
        return;
    }
    if (size == 1) {
        outputs = inputs;
        return;
    }

    const int leftSize = size / 2;
    vector<literal> leftInputs(
        inputs.begin(), inputs.begin() + leftSize
    );
    vector<literal> rightInputs(
        inputs.begin() + leftSize, inputs.end()
    );
    vector<literal> left;
    vector<literal> right;
    addExactTotalizer(formula, leftInputs, left);
    addExactTotalizer(formula, rightInputs, right);

    outputs.resize(size);
    for (int index = 0; index < size; ++index)
        outputs[index] = formula->newBoolVar();

    // If the children contain at least l and r true inputs, their parent
    // contains at least l + r true inputs.  A count of zero is a true sentinel.
    for (int leftCount = 0; leftCount <= static_cast<int>(left.size()); ++leftCount) {
        for (int rightCount = 0; rightCount <= static_cast<int>(right.size()); ++rightCount) {
            const int count = leftCount + rightCount;
            if (count == 0)
                continue;
            clause forward = outputs[count - 1];
            if (leftCount > 0)
                forward |= !left[leftCount - 1];
            if (rightCount > 0)
                forward |= !right[rightCount - 1];
            formula->addClause(forward);
        }
    }

    // The reverse clauses rule out unsupported parent thresholds.  Together
    // with the clauses above this preserves exact threshold literals even in
    // stages where overtime has zero weight or is not the active objective.
    for (int leftCount = 0; leftCount <= static_cast<int>(left.size()); ++leftCount) {
        for (int rightCount = 0; rightCount <= static_cast<int>(right.size()); ++rightCount) {
            const int count = leftCount + rightCount;
            if (count >= size)
                continue;
            clause reverse = !outputs[count];
            if (leftCount < static_cast<int>(left.size()))
                reverse |= left[leftCount];
            if (rightCount < static_cast<int>(right.size()))
                reverse |= right[rightCount];
            formula->addClause(reverse);
        }
    }
}

}  // namespace

const char *hcorapCardinalityEncodingName(
    HCORAPCardinalityEncoding encoding
) {
    switch (encoding) {
        case HCORAP_TOTALIZER:
            return "totalizer";
        case HCORAP_SORTING_NETWORK:
        default:
            return "sorting-network";
    }
}

HCORAPCardinalityEncoding parseHCORAPCardinalityEncoding(
    const string &name
) {
    if (name == "sorting-network")
        return HCORAP_SORTING_NETWORK;
    if (name == "totalizer")
        return HCORAP_TOTALIZER;
    throw invalid_argument(
        "unsupported cardinality encoding: " + name +
        " (expected sorting-network or totalizer)"
    );
}

void addHCORAPCardinalityNetwork(
    SMTFormula *formula,
    const vector<literal> &inputs,
    vector<literal> &outputs,
    HCORAPCardinalityEncoding encoding
) {
    if (encoding == HCORAP_TOTALIZER) {
        addExactTotalizer(formula, inputs, outputs);
        return;
    }
    formula->addSorting(inputs, outputs, true, true);
}
