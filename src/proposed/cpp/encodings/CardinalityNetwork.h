#ifndef HCORAP_CARDINALITY_NETWORK_H
#define HCORAP_CARDINALITY_NETWORK_H

#include "smtformula.h"

#include <string>
#include <vector>

enum HCORAPCardinalityEncoding {
    HCORAP_SORTING_NETWORK,
    HCORAP_TOTALIZER
};

const char *hcorapCardinalityEncodingName(
    HCORAPCardinalityEncoding encoding
);

HCORAPCardinalityEncoding parseHCORAPCardinalityEncoding(
    const std::string &name
);

void addHCORAPCardinalityNetwork(
    SMTFormula *formula,
    const std::vector<literal> &inputs,
    std::vector<literal> &outputs,
    HCORAPCardinalityEncoding encoding
);

#endif
