#ifndef HCORAP_IMPLIED_CONSTRAINTS_H
#define HCORAP_IMPLIED_CONSTRAINTS_H

#include <string>

enum HCORAPImpliedConfig {
    HCORAP_IMPLIED_NONE,
    HCORAP_IMPLIED_USER_SLOTS,
    HCORAP_IMPLIED_SLOT_CAPACITY,
    HCORAP_IMPLIED_BOTH,
    HCORAP_IMPLIED_BOTH_PLUS
};

const char *hcorapImpliedConfigName(HCORAPImpliedConfig config);

HCORAPImpliedConfig parseHCORAPImpliedConfig(const std::string &name);

bool hcorapUsesUserSlots(HCORAPImpliedConfig config);
bool hcorapUsesSlotCapacity(HCORAPImpliedConfig config);
bool hcorapUsesPlusImprovements(HCORAPImpliedConfig config);

#endif
