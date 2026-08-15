# Compute abstraction layer
# This module will contain hardware-aware compute primitives and configurations.

from .config import ComputeConfig, config, get_config, set_config

__all__ = ["ComputeConfig", "config", "get_config", "set_config"]
