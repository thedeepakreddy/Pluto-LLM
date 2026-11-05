"""Small helpers for publishing versioned Pluto artifacts to the Hugging Face Hub."""
from __future__ import annotations

from pathlib import Path

from llm.manifest import read_manifest, verify_files


def publish_folder(local_dir: Path, hub_repo: str, path_in_repo: str, api=None) -> None:
    """Verify a manifested artifact folder, then publish it to a private model repo."""
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise FileNotFoundError(f"artifact folder does not exist: {local_dir}")
    read_manifest(local_dir)
    verify_files(local_dir)

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(hub_repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo,
                      repo_id=hub_repo, repo_type="model")
