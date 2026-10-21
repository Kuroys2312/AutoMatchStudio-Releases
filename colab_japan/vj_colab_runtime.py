# -*- coding: utf-8 -*-
"""Build the pinned runtimes on Colab: Python 3.11.9 venvs, SBV2 source, models.

Nothing here chooses a version: everything comes from ``pins.json``, a copy
of the app's requirements and dependency profiles.  Each venv is verified
after installation; a mismatch stops the notebook instead of running with
whatever Colab happens to have.

Copyright (C) 2026 Kuroys2312.  Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from vj_colab_common import PackageError, load_pins, pins_sha256, say, sha256_file

UV_VERSION = "0.4.30"
# Resolution overrides that apply on Linux only.  The pins come from the
# Windows runtimes, where triton is never installed.  On Linux torch
# 2.5.1+cu121 requires triton==3.1.0 while openai-whisper 20231117 still
# declares triton<3 (later whisper releases dropped that bound).  Whisper only
# uses triton for its optional CUDA timing kernels.  No pinned package changes.
LINUX_OVERRIDES = {"asr": ("triton==3.1.0",)}
ENV_ROOT = Path(os.environ.get("VJ_ENV_ROOT", "/content/vj_env"))
SBV2_ROOT = Path(os.environ.get("VJ_SBV2_ROOT", "/content/vj_sbv2"))
HF_CACHE = Path(os.environ.get("VJ_HF_CACHE", "/content/vj_hf"))


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    completed = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if completed.returncode:
        tail = (completed.stdout[-3000:] + "\n" + completed.stderr[-3000:]).strip()
        raise PackageError(f"Lệnh thất bại ({command[0]} …):\n{tail}")
    return completed


def _uv() -> list[str]:
    try:
        version = run([sys.executable, "-m", "uv", "--version"]).stdout.split()[1]
    except (PackageError, IndexError):
        version = ""
    if version != UV_VERSION:
        say(f"Cài uv {UV_VERSION}…")
        run([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"])
    return [sys.executable, "-m", "uv"]


def venv_python(kind: str) -> Path:
    return ENV_ROOT / kind / "bin" / "python"


def _requirement_name(line: str) -> tuple[str, str]:
    name, _, version = line.partition("==")
    return name.strip(), version.strip()


def _installed_versions(python: Path, names: list[str]) -> dict:
    code = ("import json,sys\nfrom importlib.metadata import version, PackageNotFoundError\nout={}\n"
            "for n in json.loads(sys.argv[1]):\n    try: out[n]=version(n)\n"
            "    except PackageNotFoundError: out[n]=None\nprint(json.dumps(out))")
    return json.loads(run([str(python), "-c", code, json.dumps(names)]).stdout)


def build_venv(kind: str, pins: dict | None = None) -> Path:
    """Create (once per session) and verify the venv for ``asr`` or ``train``."""
    pins = pins or load_pins()
    spec = pins[kind]
    expected = pins_sha256(pins, kind)
    python = venv_python(kind)
    marker = ENV_ROOT / kind / ".vj_pins_sha256"
    if python.exists() and marker.is_file() and marker.read_text().strip() == expected:
        say(f"Môi trường {kind} đã có sẵn trong phiên này.")
        return python
    uv = _uv()
    say(f"Cài Python {spec['python']} (uv)…")
    run([*uv, "python", "install", spec["python"]])
    shutil.rmtree(ENV_ROOT / kind, ignore_errors=True)
    run([*uv, "venv", "--python", spec["python"], str(ENV_ROOT / kind)])
    requirements = ENV_ROOT / f"{kind}_requirements.txt"
    requirements.write_text("\n".join(spec["venv"]["requirements"]) + "\n", encoding="utf-8")
    command = [*uv, "pip", "install", "--python", str(python), "--index-strategy", "unsafe-best-match",
               "-r", str(requirements)]
    for url in spec["venv"]["index_urls"]:
        command[5:5] = ["--extra-index-url", url]
    if LINUX_OVERRIDES.get(kind) and sys.platform.startswith("linux"):
        overrides = ENV_ROOT / f"{kind}_overrides.txt"
        overrides.write_text("\n".join(LINUX_OVERRIDES[kind]) + "\n", encoding="utf-8")
        command += ["--override", str(overrides)]
    say(f"Cài {len(spec['venv']['requirements'])} thư viện đã ghim cho {kind} (vài phút)…")
    run(command)
    actual_python = run([str(python), "-c", "import platform; print(platform.python_version())"]).stdout.strip()
    if actual_python != spec["python"]:
        raise PackageError(f"Python của môi trường {kind} là {actual_python}, cần {spec['python']}.")
    wanted = dict(_requirement_name(line) for line in spec["venv"]["requirements"] if "==" in line)
    installed = _installed_versions(python, list(wanted))
    wrong = {name: (installed.get(name), version) for name, version in wanted.items() if installed.get(name) != version}
    if wrong:
        details = ", ".join(f"{name}={have} (cần {want})" for name, (have, want) in sorted(wrong.items()))
        raise PackageError(f"Thư viện cài sai phiên bản: {details}")
    marker.write_text(expected)
    say(f"✓ Môi trường {kind}: Python {actual_python}, {len(wanted)} thư viện đúng phiên bản.")
    return python


def prepare_sbv2(pins: dict | None = None) -> Path:
    """Clone Style-Bert-VITS2 at the pinned commit (unmodified)."""
    source = (pins or load_pins())["train"]["source"]
    if not (SBV2_ROOT / ".git").is_dir():
        shutil.rmtree(SBV2_ROOT, ignore_errors=True)
        say(f"Tải mã nguồn Style-Bert-VITS2 {source['tag']}…")
        run(["git", "clone", "--quiet", "--no-checkout", source["repository"], str(SBV2_ROOT)])
        run(["git", "-C", str(SBV2_ROOT), "checkout", "--quiet", "--detach", source["commit"]])
    head = run(["git", "-C", str(SBV2_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    if head != source["commit"]:
        raise PackageError(f"Mã nguồn SBV2 ở commit {head}, cần {source['commit']}.")
    if run(["git", "-C", str(SBV2_ROOT), "status", "--porcelain", "--untracked-files=no"]).stdout.strip():
        raise PackageError("Mã nguồn SBV2 bị sửa đổi; hãy khởi động lại phiên Colab.")
    return SBV2_ROOT


_PROVISION = r'''
import json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
assets, root, cache = json.loads(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out = {}
for a in assets:
    if a["cache_only"]:
        path = hf_hub_download(repo_id=a["repository"], filename=a["remote_path"], revision=a["revision"], cache_dir=str(cache))
    else:
        destination = root / a["source_relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo_id=a["repository"], filename=a["remote_path"], revision=a["revision"], local_dir=str(destination.parent))
        if Path(path).resolve() != destination.resolve():
            raise SystemExit(f"asset {a['id']} landed at {path}")
    out[a["id"]] = str(path)
print(json.dumps(out))
'''


def provision_training_assets(python: Path, pins: dict | None = None) -> dict:
    """Download the pinned base/BERT/WavLM/WeSpeaker files and verify SHA-256.

    Returns ``{asset_id: sha256}`` of the verified JP-Extra base files.
    """
    pins = pins or load_pins()
    assets = pins["train"]["assets"]
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    say("Tải model gốc JP-Extra, BERT, WavLM, WeSpeaker (kiểm SHA-256)…")
    paths = json.loads(run([str(python), "-c", _PROVISION, json.dumps(assets), str(SBV2_ROOT), str(HF_CACHE)],
                           env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1"}).stdout.strip().splitlines()[-1])
    base = {}
    for asset in assets:
        actual = sha256_file(Path(paths[asset["id"]]))
        if actual != asset["sha256"]:
            raise PackageError(f"Model {asset['id']} tải về sai mã kiểm tra ({actual}).")
        if asset["cache_only"] and asset.get("cache_reference"):
            # SBV2's style_gen.py asks pyannote for the ``main`` ref; bind it
            # to the pinned revision so the offline run uses exactly that.
            ref = HF_CACHE / ("models--" + asset["repository"].replace("/", "--")) / "refs" / asset["cache_reference"]
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(asset["revision"], encoding="utf-8")
        if asset["id"].startswith("jp_extra_base_"):
            base[asset["id"]] = actual
    say(f"✓ {len(assets)} file model đúng mã kiểm tra.")
    return base


_KOTOBA = r'''
import json, sys
from huggingface_hub import snapshot_download
files, model, revision, cache = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
print(snapshot_download(repo_id=model, revision=revision, cache_dir=cache, allow_patterns=files))
'''


def provision_kotoba(python: Path, pins: dict | None = None) -> Path:
    """Download the pinned Kotoba snapshot and verify all manifest files."""
    kotoba = (pins or load_pins())["asr"]["kotoba"]
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    say("Tải Kotoba Whisper v2.1 (kiểm SHA-256 từng file)…")
    snapshot = Path(run([str(python), "-c", _KOTOBA, json.dumps(sorted(kotoba["files"])), kotoba["model_id"],
                         kotoba["revision"], str(HF_CACHE)],
                        env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1"}).stdout.strip().splitlines()[-1])
    for relative, digest in kotoba["files"].items():
        path = snapshot / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise PackageError(f"File Kotoba {relative} thiếu hoặc sai mã kiểm tra.")
    say(f"✓ Kotoba: {len(kotoba['files'])} file đúng mã kiểm tra.")
    return snapshot


__all__ = ["HF_CACHE", "SBV2_ROOT", "build_venv", "prepare_sbv2", "provision_kotoba", "provision_training_assets",
           "venv_python"]
