"""Upload completed training artifacts and metric summaries to Hugging Face datasets repo."""

from __future__ import annotations

import logging
import os
import zipfile
from datetime import datetime
from pathlib import Path

from config.artifact_upload_config import ArtifactUploadConfig


logger = logging.getLogger(__name__)


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


def upload_artifact_if_enabled(upload_config: ArtifactUploadConfig, output_dir: str | Path) -> None:
    if not upload_config.enabled:
        logger.info("Artifact upload is disabled in configuration. Skipping HF upload.")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()

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
    zip_name = f"GW_AVF_Multimodal_{timestamp}.zip"
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

    path_in_repo = f"GW_AVF_Multimodal_{timestamp}.zip"
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
