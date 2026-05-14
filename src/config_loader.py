import logging
import yaml
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class VoiceConfig:
    """Load and parse Revenant Echo config from YAML + environment."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load YAML config and merge environment overrides."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        env_path = self.config_path.parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value by dot-notation path (e.g., 'stt.model_size')."""
        keys = path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        return self.config.get(key, {})


def setup_logging(config: VoiceConfig):
    """Configure logging based on config."""
    log_level = config.get("logging.level", "INFO")
    log_file = config.get("logging.file", "logs/revenant_echo.log")

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)
