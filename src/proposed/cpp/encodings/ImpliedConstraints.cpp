#include "ImpliedConstraints.h"

#include <stdexcept>

using namespace std;

const char *hcorapImpliedConfigName(HCORAPImpliedConfig config) {
    switch (config) {
        case HCORAP_IMPLIED_USER_SLOTS:
            return "user-slots";
        case HCORAP_IMPLIED_SLOT_CAPACITY:
            return "slot-capacity";
        case HCORAP_IMPLIED_BOTH:
            return "both";
        case HCORAP_IMPLIED_BOTH_PLUS:
            return "both-plus";
        case HCORAP_IMPLIED_NONE:
        default:
            return "none";
    }
}

HCORAPImpliedConfig parseHCORAPImpliedConfig(const string &name) {
    if (name == "none")
        return HCORAP_IMPLIED_NONE;
    if (name == "user-slots")
        return HCORAP_IMPLIED_USER_SLOTS;
    if (name == "slot-capacity")
        return HCORAP_IMPLIED_SLOT_CAPACITY;
    if (name == "both")
        return HCORAP_IMPLIED_BOTH;
    if (name == "both-plus")
        return HCORAP_IMPLIED_BOTH_PLUS;
    throw invalid_argument(
        "unsupported implied-constraints configuration: " + name +
        " (expected none, user-slots, slot-capacity, both or both-plus)"
    );
}

bool hcorapUsesUserSlots(HCORAPImpliedConfig config) {
    return config == HCORAP_IMPLIED_USER_SLOTS ||
        config == HCORAP_IMPLIED_BOTH ||
        config == HCORAP_IMPLIED_BOTH_PLUS;
}

bool hcorapUsesSlotCapacity(HCORAPImpliedConfig config) {
    return config == HCORAP_IMPLIED_SLOT_CAPACITY ||
        config == HCORAP_IMPLIED_BOTH ||
        config == HCORAP_IMPLIED_BOTH_PLUS;
}

bool hcorapUsesPlusImprovements(HCORAPImpliedConfig config) {
    return config == HCORAP_IMPLIED_BOTH_PLUS;
}
