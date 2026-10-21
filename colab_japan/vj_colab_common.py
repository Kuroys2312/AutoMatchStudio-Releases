# -*- coding: utf-8 -*-
"""Package format shared by the KuroYs Voice Japan notebook scripts.

Standard library only: this file runs in Colab's own Python before any
pinned environment exists.  It mirrors the app's format (job.json in,
result.json out, every file listed with size and SHA-256).

Copyright (C) 2026 Kuroys2312.  Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time
import zipfile

JOB_SCHEMA = "kuroys.voicejapan.colab.job"
RESULT_SCHEMA = "kuroys.voicejapan.colab.result"
SCHEMA_VERSION = 1
KINDS = ("asr", "train")
# The tag this copy of the notebook scripts is published under.
NOTEBOOK_TAG = "vj-colab-v1"

HERE = Path(__file__).resolve().parent
_PACKAGE_ID = re.compile(r"[0-9a-f]{32}")
_MAX_MEMBERS = 20_000
_MAX_TOTAL_BYTES = 4 * 1024 ** 3


class PackageError(RuntimeError):
    """The uploaded package cannot be used; the message is for the user."""


def say(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:  # a non-UTF-8 console (local tests); Colab is UTF-8
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, "replace").decode(encoding), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def safe_member_path(name: object) -> str:
    if type(name) is not str or not name or "\\" in name or ":" in name or name.startswith("/"):
        raise PackageError(f"Tên file trong gói không an toàn: {name!r}")
    if any(part in ("", ".", "..") for part in PurePosixPath(name).parts) or name.endswith("/"):
        raise PackageError(f"Tên file trong gói không an toàn: {name!r}")
    return name


def load_pins() -> dict:
    return json.loads((HERE / "pins.json").read_text(encoding="utf-8"))


def pins_sha256(pins: dict, kind: str) -> str:
    return canonical_sha256({"schema_version": pins["schema_version"], "kind": kind, "runtime": pins[kind]})


def open_job(zip_path: Path, work_root: Path) -> tuple[dict, Path]:
    """Check ``job.json`` and every listed file, extract them; return (job, folder)."""
    zip_path = Path(zip_path)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            job = json.loads(archive.read("job.json").decode("utf-8"))
            infos = archive.infolist()
            if len(infos) > _MAX_MEMBERS or sum(info.file_size for info in infos) > _MAX_TOTAL_BYTES:
                raise PackageError("Gói quá lớn bất thường.")
            if not isinstance(job, dict) or job.get("schema") != JOB_SCHEMA:
                raise PackageError("Đây không phải gói Voice Japan Colab (thiếu job.json đúng định dạng).")
            if job.get("schema_version") != SCHEMA_VERSION:
                raise PackageError("Gói tạo từ phiên bản app khác: hãy mở notebook theo nút trong app.")
            if job.get("colab_tag") != NOTEBOOK_TAG:
                raise PackageError(f"Gói dành cho notebook {job.get('colab_tag')!r}, đây là {NOTEBOOK_TAG!r}: "
                                   "hãy mở notebook bằng nút \"Mở trang Colab\" trong app.")
            if job.get("kind") not in KINDS or not _PACKAGE_ID.fullmatch(str(job.get("package_id"))):
                raise PackageError("Gói thiếu loại hoặc mã gói hợp lệ.")
            folder = Path(work_root) / job["package_id"] / "job"
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)
            for entry in job.get("files", []):
                name = safe_member_path(entry.get("path"))
                if name in ("job.json", "result.json"):
                    raise PackageError(f"Tên file trong gói không hợp lệ: {name}")
                target = folder.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if target.stat().st_size != entry.get("size") or sha256_file(target) != entry.get("sha256"):
                    raise PackageError(f"File {name} trong gói bị hỏng (mã kiểm tra không khớp): hãy tải gói lên lại.")
            (folder / "job.json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    except KeyError as exc:
        raise PackageError(f"Gói thiếu file: {exc}") from exc
    except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError("File tải lên bị hỏng hoặc không phải file .zip của app.") from exc
    pins = load_pins()
    if job["payload"].get("pins_sha256") != pins_sha256(pins, job["kind"]):
        raise PackageError("Phiên bản thư viện/model của app khác notebook này: hãy cập nhật app "
                           "và mở notebook bằng nút trong app.")
    return job, folder


def write_result(output: Path, *, kind: str, package_id: str, files: list[tuple[str, Path]],
                 payload: dict, runtime: dict, notes: list[str] | None = None) -> Path:
    """Write the result zip atomically: ``result.json`` plus listed files."""
    entries = []
    for name, source in files:
        safe_member_path(name)
        entries.append({"path": name, "size": Path(source).stat().st_size, "sha256": sha256_file(source)})
    result = {"schema": RESULT_SCHEMA, "schema_version": SCHEMA_VERSION, "kind": kind, "package_id": package_id,
              "colab_tag": NOTEBOOK_TAG, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "files": entries, "payload": payload, "runtime": runtime, "notes": list(notes or [])}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("result.json", json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            for name, source in files:
                archive.write(source, name)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def result_name(kind: str, package_id: str) -> str:
    return f"vj_colab_{kind}_result_{package_id[:8]}.zip"


def python_version_of(python: Path) -> str:
    import subprocess
    out = subprocess.run([str(python), "-c", "import platform; print(platform.python_version())"],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def gpu_name() -> str:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    except (OSError, IndexError, Exception):
        return ""


if __name__ == "__main__":  # pragma: no cover - manual check only
    say(f"vj_colab_common {NOTEBOOK_TAG} on Python {sys.version.split()[0]}")
