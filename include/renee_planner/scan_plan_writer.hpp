#ifndef RENEE_PLANNER__SCAN_PLAN_WRITER_HPP_
#define RENEE_PLANNER__SCAN_PLAN_WRITER_HPP_

#include "renee_planner/scan_types.hpp"

#include <string>

namespace renee_planner
{

class ScanPlanWriter
{
public:
  ScanPlanWriter() = default;

  void writeYamlFile(const ScanPlan & plan, const std::string & output_file) const;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__SCAN_PLAN_WRITER_HPP_
