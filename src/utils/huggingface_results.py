"""Upload final result files and complete artifact zips to Hugging Face datasets repo."""

from __future__ import annotations

import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path

from config.artifact_upload_config import ArtifactUploadConfig
from settings import ResultsUploadConfig


logger = logging.getLogger(__name__)

RESULT_ARTIFACT_FILENAMES = (
    "best_val_metrics.json",
    "best_val_confusion_matrix.csv",
    "test_metrics.json",
    "test_confusion_matrix.csv",
    "summary_results.csv",
)


def result_path_in_repo(path_prefix: str, run_name: str, timestamp: str, filename: str) -> str:
    """Build a stable, non-overwriting Hugging Face Dataset path for one CSV/JSON file."""
    clean_prefix = path_prefix.strip("/")
    return f"{clean_prefix}/{run_name}/{timestamp}/{filename}"


def _resolve_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def zip_output_directory(source_dir: str | Path, output_zip_path: str | Path) -> Path:
    source = Path(source_dir)
    target_zip = Path(output_zip_path)
    target_zip.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Zipping complete artifact directory '{source}' -> '{target_zip}'...")
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(source):
            for file in files:
                file_path = Path(root) / file
                if file_path.resolve() == target_zip.resolve():
                    continue
                arcname = file_path.relative_to(source)
                zip_file.write(file_path, arcname)
    logger.info(f"Successfully zipped artifact directory ({target_zip.stat().st_size / 1e6:.2f} MB).")
    return target_zip


def upload_result_files(config: ResultsUploadConfig, output_dir: Path | str) -> None:
    """Upload final validation/test metrics and confusion files; preserve local results on failure."""
    if not config.enabled:
        logger.info("Hugging Face result upload is disabled.")
        return
    result_files = [Path(output_dir) / filename for filename in RESULT_ARTIFACT_FILENAMES]
    existing_files = [path for path in result_files if path.is_file()]
    if not existing_files:
        logger.warning("Skipping Hugging Face result upload because no output files were found in '%s'.", output_dir)
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
        for result_file in existing_files:
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


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, output_dir: str | Path) -> None:
    if not upload_config.enabled:
        logger.info("Artifact upload is disabled in configuration. Skipping HF upload.")
        return

    token = _resolve_token()
    if not token:
        logger.warning(
            "HF_TOKEN environment variable is not set and no cached token found. "
            "Skipping automatic upload to Hugging Face repository."
        )
        return

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError:
        logger.warning("huggingface_hub library is not installed. Skipping upload.")
        return

    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"Multimodal_Artifacts_{timestamp}.zip"
    output_zip = output_path.parent / zip_name

    zipped_file = zip_output_directory(output_path, output_zip)

    if upload_config.create_repo:
        try:
            create_repo(
                repo_id=upload_config.repo_id,
                repo_type=upload_config.repo_type,
                token=token,
                exist_ok=True,
            )
        except Exception as err:
            logger.warning(f"Repo creation check warning: {err}")

    path_in_repo = f"Multimodal_Artifacts_{timestamp}.zip"
    logger.info(f"Uploading full artifact folder zip to HF Repo '{upload_config.repo_id}' at path '{path_in_repo}'...")

    try:
        upload_file(
            path_or_fileobj=str(zipped_file),
            path_in_repo=path_in_repo,
            repo_id=upload_config.repo_id,
            repo_type=upload_config.repo_type,
            token=token,
        )
        logger.info(f"Successfully uploaded full artifact zip to https://huggingface.co/datasets/{upload_config.repo_id}")
    except Exception as exc:
        logger.error(f"Failed to upload artifact to Hugging Face: {exc}")
