"""
Hardware-agnostic configuration loading and validation.
"""

from .loader import load_config, get_default_config_path

__all__ = ["load_config", "get_default_config_path"]
