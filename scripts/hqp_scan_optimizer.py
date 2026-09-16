#!/usr/bin/env python3
"""CMake-installed entry point for the standalone/ROS scan optimizer."""

from renee_trajectory_generation.scan_optimizer import main


if __name__ == '__main__':
    raise SystemExit(main())
