#ifndef HCORAP_SYMMETRY_BREAKING_H
#define HCORAP_SYMMETRY_BREAKING_H

#include <string>

enum HCORAPSymmetryBreaking {
    HCORAP_SYMMETRY_NONE,
    HCORAP_SYMMETRY_SLOTS,
    HCORAP_SYMMETRY_SERVICES,
    HCORAP_SYMMETRY_SLOT_SERVICE,
    HCORAP_SYMMETRY_ALL
};

const char *hcorapSymmetryBreakingName(HCORAPSymmetryBreaking config);

HCORAPSymmetryBreaking parseHCORAPSymmetryBreaking(const std::string &name);

bool hcorapBreaksSlotSymmetry(HCORAPSymmetryBreaking config);
bool hcorapBreaksServiceSymmetry(HCORAPSymmetryBreaking config);
bool hcorapBreaksAgentSymmetry(HCORAPSymmetryBreaking config);

#endif
