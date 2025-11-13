"""
Configuration loader for Energy Trading system.

This module provides centralized configuration loading from YAML files
and environment variables, plus logging initialization.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file. If None, uses default location.

    Returns:
        Dictionary containing configuration settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Substitute environment variables in config
    config = _substitute_env_vars(config)

    return config


def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variable placeholders in config.

    Replaces ${VAR_NAME} with the value of environment variable VAR_NAME.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with environment variables substituted
    """
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # Extract variable name and get from environment
        var_name = config[2:-1]
        return os.getenv(var_name, config)  # Keep placeholder if not found
    else:
        return config


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Initialize logging configuration from config dictionary.

    Sets up:
    - Root logger level
    - Console handler (stdout)
    - File handler with rotation (if log_file specified)
    - Formatting

    Args:
        config: Configuration dictionary. If None, loads from default location.
    """
    if config is None:
        config = load_config()

    # Get logging configuration
    log_config = config.get("logging", {})
    log_level = os.getenv("LOG_LEVEL") or log_config.get("level", "INFO")
    log_file = log_config.get("log_file", "logs/energy_trading.log")
    log_format = log_config.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")

    # Rotation settings
    rotation = log_config.get("rotation", {})
    max_bytes = rotation.get("max_bytes", 10485760)  # 10 MB default
    backup_count = rotation.get("backup_count", 5)

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (if log file specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized")


# Singleton config instance
_config_instance: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """
    Get the global configuration instance (singleton pattern).

    Loads configuration on first call and caches it for subsequent calls.

    Returns:
        Configuration dictionary
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
        setup_logging(_config_instance)
    return _config_instance


def reset_config() -> None:
    """
    Reset the global configuration singleton.

    This is useful for testing when you need to inject a test-specific config.
    After calling this, the next call to get_config() will reload from file.
    """
    global _config_instance
    _config_instance = None


if __name__ == "__main__":
    # Test configuration loading
    print("Loading configuration...")
    config = load_config()

    print("\nConfiguration loaded successfully!")
    print(f"API Base URLs:")
    print(f"  EIA: {config['api']['eia']['base_url']}")
    print(f"  CAISO: {config['api']['caiso']['base_url']}")

    print(f"\nData Paths:")
    print(f"  Raw: {config['data']['raw_data_path']}")
    print(f"  Processed: {config['data']['processed_data_path']}")

    print(f"\nLogging:")
    print(f"  Level: {config['logging']['level']}")
    print(f"  File: {config['logging']['log_file']}")

    # Test logging setup
    print("\nInitializing logging...")
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Test log message - INFO")
    logger.debug("Test log message - DEBUG")
    logger.warning("Test log message - WARNING")

    print("\nLogging initialized successfully!")
