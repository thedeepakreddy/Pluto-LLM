"""Run persistence to the Hugging Face Hub, tested against an in-memory fake Hub
(huggingface.co is not reachable from CI)."""
import fnmatch
import shutil
import sys
import types

import pytest

from conftest import run_pipeline
from llm import cli
from llm.hub import publish_folder
from llm.manifest import ProvenanceError, write_manifest
from llm.runs import create_run
from llm.train.loop import train


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    remote = tmp_path / "remote_hub"
    calls = []

    class HfApi:
        def create_repo(self, repo_id, private=True, exist_ok=True, **kw):
            calls.append(("create_repo", repo_id, private))
            (remote / repo_id).mkdir(parents=True, exist_ok=True)

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
            calls.append(("upload_file", path_in_repo))
            dest = remote / repo_id / path_in_repo
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path_or_fileobj, dest)

        def upload_folder(self, folder_path, path_in_repo, repo_id, ignore_patterns=(), **kw):
            calls.append(("upload_folder", path_in_repo))
            from pathlib import Path
            for f in Path(folder_path).rglob("*"):
                rel = f.relative_to(folder_path).as_posix()
                if f.is_file() and not any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
                    dest = remote / repo_id / path_in_repo / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, dest)

    def snapshot_download(repo_id, allow_patterns, local_dir, **kw):
        base = remote / repo_id
        for f in base.rglob("*"):
            rel = f.relative_to(base).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                dest = __import__("pathlib").Path(local_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest)
        return str(local_dir)

    mod = types.SimpleNamespace(HfApi=HfApi, snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return remote, calls


def test_run_dir_survives_session_death(tiny_config, workspace, fake_hub):
    remote, calls = fake_hub
    cfg, _ = run_pipeline(tiny_config, ["train.hub_repo=me/pluto-runs"])
    run = create_run(cfg)
    train(cfg, run, stop_at_step=15)
    run_id = run.name
    assert (remote / "me/pluto-runs" / run_id / "config.yaml").exists()
    assert (remote / "me/pluto-runs" / run_id / "checkpoints" / "latest.json").exists()
    assert ("create_repo", "me/pluto-runs", True) in calls  # private by default

    shutil.rmtree(workspace / "runs")  # the session dies; /kaggle/working is gone
    assert cli.main(["pull-run", "me/pluto-runs", run_id]) == 0
    assert cli.main(["resume", str(workspace / "runs" / run_id)]) == 0
    resumed = (workspace / "runs" / run_id / "resume_log.jsonl").read_text()
    assert '"step": 15' in resumed


def test_no_hub_calls_without_hub_repo(prepared_tokenized, fake_hub):
    cfg, _ = prepared_tokenized
    train(cfg, create_run(cfg), stop_at_step=3)
    assert fake_hub[1] == []


class RecordingHubApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, **kwargs):
        self.created.append((repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


def test_publish_folder_creates_private_repo_and_uploads_verified_artifact(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text('{"vocab": []}')
    write_manifest(tokenizer_dir, "tokenizer", "m3-bpe", 1, ["tokenizer.json"], {})
    api = RecordingHubApi()

    publish_folder(tokenizer_dir, "user/pluto-runs", "tokenizers/m3-bpe-v1", api=api)

    assert api.created == [("user/pluto-runs", {"private": True, "exist_ok": True})]
    assert api.uploaded == [{
        "folder_path": str(tokenizer_dir),
        "path_in_repo": "tokenizers/m3-bpe-v1",
        "repo_id": "user/pluto-runs",
        "repo_type": "model",
    }]


def test_publish_folder_rejects_artifact_without_manifest(tmp_path):
    (tmp_path / "tokenizer.json").write_text('{"vocab": []}')

    with pytest.raises(ProvenanceError, match="no manifest.json"):
        publish_folder(tmp_path, "user/pluto-runs", "tokenizers/m3-bpe-v1", api=RecordingHubApi())
