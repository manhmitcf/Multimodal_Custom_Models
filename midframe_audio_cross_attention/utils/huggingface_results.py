"""Upload final result files to a Hugging Face Dataset without storing tokens."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from settings import ResultsUploadConfig


logger = logging.getLogger(__name__)

RESULT_ARTIFACT_FILENAMES = (
    "best_val_metrics.json",
    "best_val_confusion_matrix.csv",
    "test_metrics.json",
    "test_confusion_matrix.csv",
)


def result_path_in_repo(path_prefix: str, run_name: str, timestamp: str, filename: str) -> str:
    """Build a stable, non-overwriting Hugging Face Dataset path for one CSV."""
    clean_prefix = path_prefix.strip("/")
    return f"{clean_prefix}/{run_name}/{timestamp}/{filename}"


def _resolve_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def upload_result_files(config: ResultsUploadConfig, output_dir: Path) -> None:
    """Upload final validation/test metrics and confusion files; preserve local results on failure."""
    if not config.enabled:
        logger.info("Hugging Face result upload is disabled.")
        return
    result_files = [Path(output_dir) / filename for filename in RESULT_ARTIFACT_FILENAMES]
    missing = [str(path) for path in result_files if not path.is_file()]
    if missing:
        logger.warning("Skipping Hugging Face result upload because final output files are missing: %s", missing)
        return
    token = _resolve_token()
    if not token:
        logger.warning("Skipping Hugging Face result upload: set HF_TOKEN or run 'hf auth login' on Marimo.")
        return
    try:
        from huggingface_hub import create_repo, upload_file

        if config.create_repo:
            create_repo(
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                token=token,
                exist_ok=True,
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = Path(output_dir).name
        for result_file in result_files:
            destination = result_path_in_repo(config.path_prefix, run_name, timestamp, result_file.name)
            upload_file(
                path_or_fileobj=str(result_file),
                path_in_repo=destination,
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                token=token,
            )
            logger.info("Uploaded result file to Hugging Face: %s", destination)
    except Exception:
        logger.exception("Hugging Face result upload failed; local result files remain in '%s'.", output_dir)
