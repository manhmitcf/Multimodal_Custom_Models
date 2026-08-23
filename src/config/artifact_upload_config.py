"""Configuration for uploading completed training artifacts to Hugging Face repository."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ArtifactUploadConfig:
    enabled: bool = True
    source_dir: str = "checkpoint"
    zip_path: str = "checkpoint/gw_avf_multimodal_results.zip"
    repo_id: str = "manhmitcf/fish_result"
    repo_type: str = "dataset"
    path_in_repo: str = "gw_avf_multimodal_results.zip"
    create_repo: bool = True

    @classmethod
    def from_json(cls, path: str | Path = "config/artifact_upload_config.json") -> "ArtifactUploadConfig":
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Upload config file not found at '{path}', using default configuration.")
            return cls()
        logger.info(f"Loading artifact upload configuration from JSON: '{path}'")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
