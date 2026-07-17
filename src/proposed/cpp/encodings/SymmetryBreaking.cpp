#include "SymmetryBreaking.h"

#include <stdexcept>

using namespace std;

const char *hcorapSymmetryBreakingName(HCORAPSymmetryBreaking config) {
    switch (config) {
        case HCORAP_SYMMETRY_SLOTS:
            return "slots";
        case HCORAP_SYMMETRY_SERVICES:
            return "services";
        case HCORAP_SYMMETRY_SLOT_SERVICE:
            return "slot-service";
        case HCORAP_SYMMETRY_ALL:
            return "all";
        case HCORAP_SYMMETRY_NONE:
        default:
            return "none";
    }
}

HCORAPSymmetryBreaking parseHCORAPSymmetryBreaking(const string &name) {
    if (name == "none")
        return HCORAP_SYMMETRY_NONE;
    if (name == "slots")
        return HCORAP_SYMMETRY_SLOTS;
    if (name == "services")
        return HCORAP_SYMMETRY_SERVICES;
    if (name == "slot-service")
        return HCORAP_SYMMETRY_SLOT_SERVICE;
    if (name == "all")
        return HCORAP_SYMMETRY_ALL;
    throw invalid_argument(
        "unsupported symmetry-breaking configuration: " + name +
        " (expected none, slots, services, slot-service or all)"
    );
}

bool hcorapBreaksSlotSymmetry(HCORAPSymmetryBreaking config) {
    return config == HCORAP_SYMMETRY_SLOTS ||
        config == HCORAP_SYMMETRY_SLOT_SERVICE ||
        config == HCORAP_SYMMETRY_ALL;
}

bool hcorapBreaksServiceSymmetry(HCORAPSymmetryBreaking config) {
    return config == HCORAP_SYMMETRY_SERVICES ||
        config == HCORAP_SYMMETRY_SLOT_SERVICE ||
        config == HCORAP_SYMMETRY_ALL;
}

bool hcorapBreaksAgentSymmetry(HCORAPSymmetryBreaking config) {
    return config == HCORAP_SYMMETRY_ALL;
}
