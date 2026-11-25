"""Configuration loading, .env support, and versioned config updates with rollback."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ANSI_ESC_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_secret(val: str) -> str:
    cleaned = _ANSI_ESC_RE.sub("", val or "")
    cleaned = _CTRL_RE.sub("", cleaned)
    return cleaned.strip()


def get_project_root() -> Path:
    env_root = os.environ.get("AI_DAILY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # src/ai_daily/config.py -> repo root is 2 parents above src/
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), here.parents[2]]:
        if (candidate / "config").is_dir():
            return candidate
    return Path.cwd()


def load_dotenv(dotenv_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from .env without overriding already-set environment variables."""
    root = get_project_root()
    path = dotenv_path or (root / ".env")
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (len(value) >= 2) and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]
        value = sanitize_secret(value)
        loaded[key] = value
        if key not in os.environ or not os.environ[key]:
            os.environ[key] = value
    return loaded


class ConfigManager:
    """Manages JSON configs under config/ with timestamped history backups and rollback."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or get_project_root()).resolve()
        self.config_dir = self.root / "config"
        self.history_dir = self.config_dir / "history"
        self.data_dir = self.root / "data"
        self.output_dir = self.root / "output"
        load_dotenv(self.root / ".env")

    def _path(self, name: str) -> Path:
        filename = name if name.endswith(".json") else f"{name}.json"
        return self.config_dir / filename

    def load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_llm(self) -> dict[str, Any]:
        cfg = self.load("llm")
        # Allow env overrides so .env and GitHub Secrets work seamlessly
        if os.environ.get("LLM_BASE_URL"):
            cfg["base_url"] = os.environ["LLM_BASE_URL"].strip().rstrip("/")
        if os.environ.get("LLM_MODEL"):
            cfg["primary_model"] = os.environ["LLM_MODEL"].strip()
        return cfg

    def load_profile(self) -> dict[str, Any]:
        return self.load("profile")

    def load_sources(self) -> dict[str, Any]:
        return self.load("sources")

    def load_prompts(self) -> dict[str, Any]:
        return self.load("prompts")

    def resolve_api_key(self, llm_cfg: dict[str, Any] | None = None) -> str:
        cfg = llm_cfg or self.load_llm()
        env_names = cfg.get("api_key_env") or ["LLM_API_KEY", "DEEPSEEK_API_KEY"]
        for name in env_names:
            val = os.environ.get(name, "").strip()
            if val and not val.startswith("sk-your-"):
                return val
        return ""

    def save_with_backup(
        self, name: str, data: dict[str, Any], tag: str = "evolve"
    ) -> Path | None:
        """Atomically write updated config after saving a timestamped backup."""
        path = self._path(name)
        backup_path: Path | None = None
        if path.is_file():
            self.history_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            stem = path.stem
            backup_path = self.history_dir / f"{stem}.{ts}.{tag}.json"
            shutil.copy2(path, backup_path)

        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return backup_path

    def list_backups(self, name: str) -> list[Path]:
        if not self.history_dir.is_dir():
            return []
        stem = name[:-5] if name.endswith(".json") else name
        return sorted(self.history_dir.glob(f"{stem}.*.json"), reverse=True)

    def rollback(self, name: str, backup_file: str | None = None) -> Path:
        backups = self.list_backups(name)
        if not backups:
            raise FileNotFoundError(f"No backups found for config '{name}'")
        chosen = (
            (self.history_dir / backup_file) if backup_file else backups[0]
        )
        if not chosen.is_file():
            raise FileNotFoundError(f"Backup file not found: {chosen}")
        data = json.loads(chosen.read_text(encoding="utf-8"))
        self.save_with_backup(name, data, tag="pre-rollback")
        return chosen
