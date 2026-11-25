"""Self-iterating LLM model registry: discovers newer models, probes health, and upgrades safely."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ai_daily.config import ConfigManager
from ai_daily.llm.client import OpenAICompatClient
from ai_daily.models import ModelProbeResult, utc_now_iso
from ai_daily.storage import StateStore

_VERSION_RE = re.compile(r"(?:^|[-_v])(\d+)(?:\.(\d+))?(?:\.(\d+))?")
_DATE8_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_DATE4_RE = re.compile(r"(?:flash|chat|reasoner|preview|exp|r\d|v\d+(?:\.\d+)?)[-_]?(\d{4})(?:\b|$)", re.IGNORECASE)


def parse_model_version(model_id: str) -> tuple[int, int, int, int, int, str]:
    """Produce a comparable tuple so newer DeepSeek/LLM snapshots sort above older ones.

    Example:
        deepseek-v4.1-flash1017  -> (1, 4, 1, 0, 1017, 'deepseek-v4.1-flash1017')
        deepseek-v4.2-flash1120  -> (1, 4, 2, 0, 1120, 'deepseek-v4.2-flash1120')
        deepseek-v4.1-flash1118  -> (1, 4, 1, 0, 1118, 'deepseek-v4.1-flash1118')
    """
    low = model_id.lower().strip()
    is_flash = 1 if "flash" in low else 0

    major = minor = patch = 0
    # Prefer explicit vX.Y pattern first (e.g. v4.1, v3.2)
    v_match = re.search(r"v(\d+)(?:\.(\d+))?(?:\.(\d+))?", low)
    if v_match:
        major = int(v_match.group(1))
        minor = int(v_match.group(2) or 0)
        patch = int(v_match.group(3) or 0)
    else:
        m = _VERSION_RE.search(low)
        if m:
            num = int(m.group(1))
            if num < 100:  # avoid treating a 4-digit year/date as major version
                major = num
                minor = int(m.group(2) or 0)
                patch = int(m.group(3) or 0)

    date_tag = 0
    d8 = _DATE8_RE.search(low)
    if d8:
        date_tag = int(f"{d8.group(1)}{d8.group(2)}{d8.group(3)}")
    else:
        d4 = _DATE4_RE.search(low)
        if d4:
            mmdd = int(d4.group(1))
            mm, dd = divmod(mmdd, 100)
            if 1 <= mm <= 12 and 1 <= dd <= 31:
                date_tag = mmdd

    return (is_flash, major, minor, patch, date_tag, low)


def filter_and_rank_models(
    remote_models: list[str],
    family_patterns: list[str],
    exclude_patterns: list[str],
) -> list[str]:
    """Filter available models by configured family/exclude regexes and rank newest first."""
    compiled_family = [re.compile(p, re.IGNORECASE) for p in family_patterns]
    compiled_exclude = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]

    matched: list[str] = []
    for mid in remote_models:
        if any(rx.search(mid) for rx in compiled_exclude):
            continue
        if compiled_family and not any(rx.search(mid) for rx in compiled_family):
            continue
        matched.append(mid)

    # Deduplicate while ranking newest first
    unique = list(dict.fromkeys(matched))
    unique.sort(key=parse_model_version, reverse=True)
    return unique


class ModelRegistry:
    """Discovers remote models, probes top candidates, and optionally upgrades config/llm.json."""

    def __init__(
        self,
        config_mgr: ConfigManager,
        client: OpenAICompatClient,
        store: StateStore | None = None,
    ) -> None:
        self.config_mgr = config_mgr
        self.client = client
        self.store = store
        self.registry_path: Path = config_mgr.data_dir / "model_registry.json"

    def probe_model(self, model_id: str, prompt: str = "Reply with exactly: OK") -> ModelProbeResult:
        try:
            resp = self.client.chat(
                [{"role": "user", "content": prompt}],
                model=model_id,
                temperature=0.0,
                max_tokens=16,
                json_mode=False,
            )
            return ModelProbeResult(
                model_id=model_id,
                ok=bool(resp.content.strip()),
                latency_ms=resp.latency_ms,
                version_key=parse_model_version(model_id),
            )
        except Exception as exc:
            return ModelProbeResult(
                model_id=model_id,
                ok=False,
                error=str(exc),
                version_key=parse_model_version(model_id),
            )

    def sync(self, *, apply: bool = False, probe: bool = True) -> dict[str, Any]:
        """Query /models, rank newer candidates, probe health, and optionally upgrade primary_model."""
        llm_cfg = self.config_mgr.load_llm()
        auto_cfg = llm_cfg.get("auto_iteration") or {}
        current_primary = llm_cfg["primary_model"]

        remote_models = self.client.list_models()
        ranked = filter_and_rank_models(
            remote_models,
            auto_cfg.get("family_patterns") or ["deepseek"],
            auto_cfg.get("exclude_patterns") or ["embed", "rerank", "tts", "whisper"],
        )

        max_probes = int(auto_cfg.get("max_probe_candidates", 3))
        probe_prompt = auto_cfg.get("probe_prompt", "Reply with exactly: OK")
        probe_results: list[ModelProbeResult] = []

        if probe:
            probe_targets = ranked[:max_probes]
            if current_primary in remote_models and current_primary not in probe_targets:
                probe_targets.append(current_primary)
            for mid in probe_targets:
                probe_results.append(self.probe_model(mid, prompt=probe_prompt))

        healthy_candidates = (
            [p.model_id for p in probe_results if p.ok]
            if probe
            else list(ranked)
        )
        recommended: str | None = None
        current_key = parse_model_version(current_primary)
        for cand in healthy_candidates:
            if cand != current_primary and parse_model_version(cand) > current_key:
                recommended = cand
                break

        upgraded = False
        backup_path = ""
        should_apply = apply or bool(auto_cfg.get("auto_upgrade_primary", False))
        if recommended and should_apply:
            # Update persisted llm.json (preserving old primary at front of fallback_models)
            disk_cfg = self.config_mgr.load("llm")
            old_primary = disk_cfg.get("primary_model", current_primary)
            fallbacks = [m for m in disk_cfg.get("fallback_models", []) if m != recommended]
            if old_primary and old_primary != recommended and old_primary not in fallbacks:
                fallbacks.insert(0, old_primary)
            disk_cfg["primary_model"] = recommended
            disk_cfg["fallback_models"] = fallbacks[:5]
            bpath = self.config_mgr.save_with_backup("llm", disk_cfg, tag="model-upgrade")
            backup_path = str(bpath) if bpath else ""
            upgraded = True
            if self.store:
                self.store.record_evolution_log(
                    kind="model_upgrade",
                    summary=f"Auto-upgraded primary model: {old_primary} -> {recommended}",
                    backup_path=backup_path,
                    diff_obj={"old_primary": old_primary, "new_primary": recommended},
                )

        env_override_warning = ""
        env_model = os.environ.get("LLM_MODEL", "").strip()
        if upgraded and env_model and env_model != recommended:
            env_override_warning = (
                f"检测到环境变量 LLM_MODEL={env_model} 正在覆盖 config/llm.json；"
                f"请注释掉 .env 或 Secrets 中的 LLM_MODEL，以便自动升级后的 {recommended} 生效。"
            )

        report = {
            "checked_at": utc_now_iso(),
            "base_url": self.client.base_url,
            "current_primary": current_primary,
            "remote_count": len(remote_models),
            "ranked_candidates": ranked[:10],
            "probed": [
                {
                    "model_id": p.model_id,
                    "ok": p.ok,
                    "latency_ms": p.latency_ms,
                    "error": p.error,
                }
                for p in probe_results
            ],
            "recommended_upgrade": recommended,
            "upgraded": upgraded,
            "backup_path": backup_path,
            "env_override_warning": env_override_warning,
        }

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
