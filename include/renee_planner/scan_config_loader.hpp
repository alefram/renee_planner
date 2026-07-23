#ifndef RENEE_PLANNER__SCAN_CONFIG_LOADER_HPP_
#define RENEE_PLANNER__SCAN_CONFIG_LOADER_HPP_

#include "renee_planner/scan_types.hpp"

#include <string>

namespace renee_planner
{

class ScanConfigLoader
{
public:
  ScanConfigLoader() = default;

  ScanConfig loadFromYamlFile(const std::string & config_file) const;
};

}  // namespace renee_planner

#endif  // RENEE_PLANNER__SCAN_CONFIG_LOADER_HPP_
