#ifndef RENEE_PLANNER__SCAN_PLAN_GENERATOR_HPP_
#define RENEE_PLANNER__SCAN_PLAN_GENERATOR_HPP_

#include "renee_planner/scan_types.hpp"

namespace renee_planner
{

class ScanPlanGenerator
{
public:
  ScanPlanGenerator() = default;

  ScanPlan generateInspectionPath(const ScanConfig & config) const;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__SCAN_PLAN_GENERATOR_HPP_
