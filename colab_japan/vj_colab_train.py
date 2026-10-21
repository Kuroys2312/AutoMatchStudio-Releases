# -*- coding: utf-8 -*-
"""Trip 2: train a Style-Bert-VITS2 JP-Extra voice on Colab.

Runs the unmodified upstream scripts at the pinned commit with exactly the
inputs the app's local trainer would use (same train list, same config
changes, same commands).  Checkpoints are copied to Google Drive now and
then, so a disconnected session can continue from the last copy.  The result
zip carries only the inference model (safetensors + config + style vectors)
and the facts the app checks before trusting it.

Copyright (C) 2026 Kuroys2312.  Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vj_colab_common import (PackageError, canonical_sha256, gpu_name, load_pins, result_name,  # noqa: E402
                             say, sha256_file, write_result)

WORK_ROOT = Path(os.environ.get("VJ_WORK_ROOT", "/content/vj_work"))
# Copying a ~1.3 GB checkpoint group to Drive at most this often keeps Drive
# usage and the time spent copying small.
MIRROR_MIN_SECONDS = int(os.environ.get("VJ_MIRROR_MIN_SECONDS", "600"))
_GROUP = ("G", "D", "WD")


def _env(hf_cache: Path) -> dict:
    # Same controls as the app's local trainer (offline, pinned caches).
    return {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(hf_cache), "HUGGINGFACE_HUB_CACHE": str(hf_cache), "PYANNOTE_CACHE": str(hf_cache),
            "PYTHONIOENCODING": "utf-8", "USE_LIBUV": "0"}


def prep_identity(job: dict, job_dir: Path) -> str:
    payload = job["payload"]
    return canonical_sha256({"run_id": payload["run_id"], "train_list": payload["train_list"],
                             "train_config": payload["train_config"], "pins": payload["pins_sha256"],
                             "segments": sorted((entry["path"], entry["sha256"]) for entry in job["files"])})


def _checkpoint_steps(models: Path) -> list[int]:
    """Steps that have a complete G/D/WD .pth group."""
    groups: dict[int, set] = {}
    for path in models.glob("*.pth"):
        prefix, _, step = path.stem.partition("_")
        if prefix in _GROUP and step.isdigit():
            groups.setdefault(int(step), set()).add(prefix)
    return sorted(step for step, prefixes in groups.items() if prefixes >= set(_GROUP))


# -- workspace ---------------------------------------------------------------
def build_workspace(job: dict, job_dir: Path, work: Path, source: Path) -> None:
    """The same layout and config the app's ``prepare_run`` creates."""
    payload = job["payload"]
    shutil.rmtree(work, ignore_errors=True)
    for name in ("raw", "wavs", "models"):
        (work / name).mkdir(parents=True)
    for entry in job["files"]:
        name = Path(entry["path"]).name
        shutil.copyfile(job_dir / entry["path"], work / "raw" / name)
        shutil.copyfile(job_dir / entry["path"], work / "wavs" / name)
    (work / "esd.list").write_text("".join(line + "\n" for line in payload["train_list"]), encoding="utf-8")
    config = json.loads((source / "configs/config_jp_extra.json").read_text(encoding="utf-8"))
    config["model_name"] = payload["model_name"]
    config["data"]["training_files"] = str(work / "train.list")
    config["data"]["validation_files"] = str(work / "val.list")
    config["train"].update(payload["train_config"])
    config["model"]["slm"]["model"] = str(source / "slm/wavlm-base-plus")
    (work / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_base_models(work, source)


def copy_base_models(work: Path, source: Path) -> None:
    for prefix in _GROUP:
        shutil.copyfile(source / "pretrained_jp_extra" / f"{prefix}_0.safetensors", work / "models" / f"{prefix}_0.safetensors")


def preprocess(python: Path, work: Path, source: Path, env: dict) -> None:
    commands = [
        ("Chuẩn hoá chữ", 1, [str(python), "preprocess_text.py", "--transcription-path", str(work / "esd.list"),
                              "--cleaned-path", str(work / "esd.list.cleaned"), "--train-path", str(work / "train.list"),
                              "--val-path", str(work / "val.list"), "--config-path", str(work / "config.json"),
                              "--val-per-lang", "0", "--max-val-total", "0", "--use_jp_extra", "--correct_path"]),
        ("Đặc trưng BERT", 2, [str(python), "bert_gen.py", "-c", str(work / "config.json")]),
        ("Vector style", 1, [str(python), "style_gen.py", "-c", str(work / "config.json"), "--num_processes", "1"]),
    ]
    wavs = sorted((work / "wavs").glob("*.wav"))
    for label, attempts, command in commands:
        for attempt in range(1, attempts + 1):
            say(f"Tiền xử lý: {label}…")
            completed = subprocess.run(command, cwd=source, env=env, text=True, capture_output=True,
                                       encoding="utf-8", errors="replace")
            missing = []
            if not completed.returncode:
                if "BERT" in label:
                    missing = [w.name for w in wavs if not (work / "wavs" / f"{w.stem}.bert.pt").is_file()]
                elif "style" in label:
                    missing = [w.name for w in wavs if not (work / "wavs" / f"{w.name}.npy").is_file()]
                if not missing:
                    break
            if attempt < attempts:
                for partial in (work / "wavs").glob("*.bert.pt"):
                    partial.unlink()
                continue
            detail = completed.stderr[-3000:] or completed.stdout[-3000:]
            if missing:
                detail = f"thiếu đầu ra cho {len(missing)} file: {', '.join(missing[:5])}"
            raise PackageError(f"Tiền xử lý thất bại ở bước {label}:\n{detail}")


# -- Google Drive ---------------------------------------------------------------
class DriveMirror:
    """Copy the newest checkpoint group to Drive, at most every MIRROR_MIN_SECONDS."""

    def __init__(self, drive: Path, models: Path, identity: str) -> None:
        self.drive, self.models, self.identity = drive, models, identity
        self.last_copy = time.time()
        self.copied_step = None
        self._stop = threading.Event()
        self._sizes: dict = {}
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _stable(self, step: int) -> bool:
        sizes = tuple((self.models / f"{p}_{step}.pth").stat().st_size for p in _GROUP)
        stable = self._sizes.get(step) == sizes
        self._sizes[step] = sizes
        return stable

    def _loop(self) -> None:
        while not self._stop.wait(20):
            try:
                steps = _checkpoint_steps(self.models)
                if not steps or steps[-1] == self.copied_step or time.time() - self.last_copy < MIRROR_MIN_SECONDS:
                    continue
                if self._stable(steps[-1]):
                    self.copy(steps[-1])
            except OSError:
                continue  # upstream may be rotating files; try again next poll

    def copy(self, step: int) -> None:
        target = self.drive / "ckpt"
        staging = self.drive / "ckpt.partial"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for prefix in _GROUP:
            shutil.copyfile(self.models / f"{prefix}_{step}.pth", staging / f"{prefix}_{step}.pth")
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        (self.drive / "state.json").write_text(json.dumps({"prep_identity": self.identity, "step": step}),
                                               encoding="utf-8")
        self.copied_step, self.last_copy = step, time.time()
        say(f"💾 Đã lưu checkpoint bước {step} lên Google Drive.")


def save_prep(work: Path, drive: Path, identity: str) -> None:
    drive.mkdir(parents=True, exist_ok=True)
    partial = drive / "prep.tar.partial"
    with tarfile.open(partial, "w") as archive:
        for name in ("wavs", "raw", "esd.list", "esd.list.cleaned", "train.list", "val.list", "config.json"):
            if (work / name).exists():
                archive.add(work / name, arcname=name)
    partial.replace(drive / "prep.tar")
    state = drive / "state.json"
    if not state.is_file():
        state.write_text(json.dumps({"prep_identity": identity, "step": None}), encoding="utf-8")


def restore(work: Path, drive: Path, identity: str, source: Path) -> int | None:
    """Rebuild the workspace from Drive; return the restored checkpoint step."""
    try:
        state = json.loads((drive / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if state.get("prep_identity") != identity or not (drive / "prep.tar").is_file():
        return None
    shutil.rmtree(work, ignore_errors=True)
    (work / "models").mkdir(parents=True)
    with tarfile.open(drive / "prep.tar") as archive:
        for member in archive.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts or not (member.isfile() or member.isdir()):
                raise PackageError("Dữ liệu tạm trên Drive bị hỏng; xoá thư mục gói trên Drive rồi chạy lại.")
        try:
            archive.extractall(work, filter="data")
        except TypeError:  # Python without tar extraction filters
            archive.extractall(work)
    copy_base_models(work, source)
    step = state.get("step")
    if isinstance(step, int) and all((drive / "ckpt" / f"{p}_{step}.pth").is_file() for p in _GROUP):
        for prefix in _GROUP:
            shutil.copyfile(drive / "ckpt" / f"{prefix}_{step}.pth", work / "models" / f"{prefix}_{step}.pth")
        return step
    return 0


# -- training -----------------------------------------------------------------------
def train(python: Path, work: Path, source: Path, env: dict, entrypoint: str, total_steps: int, log: Path) -> None:
    command = [str(python), entrypoint, "-c", str(work / "config.json"), "-m", str(work)]
    say(f"Bắt đầu huấn luyện ({total_steps} bước). Tiến độ in ra mỗi ~30 giây.")
    last_print = 0.0
    tail = ""
    with log.open("ab") as output:
        process = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read1(65536) if hasattr(process.stdout, "read1") else process.stdout.read(4096)
            if not chunk:
                break
            output.write(chunk)
            tail = (tail + chunk.decode("utf-8", "replace"))[-4000:]
            if time.time() - last_print > 30:
                lines = [line.strip() for line in tail.replace("\r", "\n").splitlines() if line.strip()]
                if lines:
                    say("   " + lines[-1][:200])
                last_print = time.time()
        code = process.wait()
    if code:
        raise PackageError(f"Huấn luyện dừng với mã lỗi {code}. Cuối nhật ký:\n{tail[-3000:]}")


_FACTS = r'''
import hashlib, json, math, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util.tensor_util import make_ndarray
work, export_dir, base = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
models = work / "models"
exports = sorted(export_dir.glob("*_s*.safetensors"), key=lambda p: int(p.stem.rsplit("_s", 1)[1]))
if not exports:
    raise SystemExit("upstream produced no inference export")
export = exports[-1]
step = int(export.stem.rsplit("_s", 1)[1])
checkpoint = models / f"G_{step}.pth"
if not checkpoint.is_file():
    raise SystemExit(f"no G checkpoint for exported step {step}")
trained = torch.load(checkpoint, map_location="cpu", weights_only=False).get("model", {})
exported = load_file(str(export), device="cpu")
start = load_file(str(base), device="cpu")
def d(x): return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
shared = sorted(set(trained) & set(exported))
if not shared or d(trained[shared[0]]) != d(exported[shared[0]]):
    raise SystemExit("inference export does not match its checkpoint")
keys = sorted(k for k in set(start) & set(trained) if not ("running_" in k or k.endswith("num_batches_tracked")))
if not keys or any(tuple(start[k].shape) != tuple(trained[k].shape) or str(start[k].dtype) != str(trained[k].dtype) for k in keys):
    raise SystemExit("trained generator is not comparable with the pinned base")
def aggregate(values):
    out = hashlib.sha256()
    for key in keys:
        value = values[key]
        out.update(key.encode()); out.update(str(value.dtype).encode()); out.update(repr(tuple(value.shape)).encode())
        out.update(value.detach().contiguous().numpy().tobytes())
    return out.hexdigest()
changed = next((k for k in keys if d(start[k]) != d(trained[k])), None); key = changed or keys[0]
losses, bad = [], []
for path in sorted(models.rglob("events.out.tfevents.*")):
    for event in EventFileLoader(str(path)).Load():
        if not event.HasField("summary"):
            continue
        for value in event.summary.value:
            if not value.tag.startswith("loss/"):
                continue
            if value.HasField("simple_value"):
                scalar = float(value.simple_value)
            elif value.HasField("tensor"):
                array = make_ndarray(value.tensor)
                if array.size != 1:
                    continue
                scalar = float(array.reshape(-1)[0])
            else:
                continue
            (losses if math.isfinite(scalar) else bad).append(format(scalar, ".17g"))
print(json.dumps({"export": str(export), "step": step, "losses": losses, "nonfinite": bad,
                  "weight": {"tensor_key": key, "tensor_before": d(start[key]), "tensor_after": d(trained[key]),
                             "dtype": str(start[key].dtype), "shape": list(start[key].shape),
                             "aggregate_before": aggregate(start), "aggregate_after": aggregate(trained)},
                  "torch": torch.__version__, "cuda": torch.version.cuda}))
'''


def collect_facts(python: Path, work: Path, export_dir: Path, source: Path) -> dict:
    completed = subprocess.run([str(python), "-c", _FACTS, str(work), str(export_dir),
                                str(source / "pretrained_jp_extra" / "G_0.safetensors")],
                               cwd=source, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise PackageError("Không kiểm được model sau huấn luyện:\n" + completed.stderr[-3000:])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_job(job: dict, job_dir: Path, output_dir: Path, drive_root: Path | None = None) -> Path:
    from vj_colab_runtime import HF_CACHE, build_venv, prepare_sbv2, provision_training_assets

    payload = job["payload"]
    pins = load_pins()
    python = build_venv("train", pins)
    source = prepare_sbv2(pins)
    base = provision_training_assets(python, pins)
    env = _env(HF_CACHE)
    work = WORK_ROOT / job["package_id"]
    drive = Path(drive_root) / "train" / job["package_id"] if drive_root else None
    identity = prep_identity(job, job_dir)
    resumed = restore(work, drive, identity, source) if drive else None
    if resumed is None:
        build_workspace(job, job_dir, work, source)
        preprocess(python, work, source, env)
        if drive:
            save_prep(work, drive, identity)
    elif resumed:
        say(f"↻ Tiếp tục từ checkpoint bước {resumed} đã lưu trên Google Drive.")
    else:
        say("↻ Dùng lại dữ liệu đã tiền xử lý trên Google Drive.")
    export_dir = source / "model_assets" / payload["model_name"]
    shutil.rmtree(export_dir, ignore_errors=True)
    log = work / "train_log.txt"
    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mirror = DriveMirror(drive, work / "models", identity) if drive else None
    if mirror:
        mirror.start()
    try:
        train(python, work, source, env, pins["train"]["entrypoint"], int(payload.get("total_steps") or 0), log)
    finally:
        if mirror:
            mirror.stop()
    facts = collect_facts(python, work, export_dir, source)
    if facts["nonfinite"]:
        raise PackageError(f"Loss có giá trị NaN/Inf ({len(facts['nonfinite'])} lần): huấn luyện hỏng, không xuất model.")
    if not facts["losses"]:
        raise PackageError("Không đọc được loss nào từ nhật ký huấn luyện.")
    if facts["weight"]["tensor_before"] == facts["weight"]["tensor_after"]:
        raise PackageError("Trọng số model không thay đổi sau huấn luyện.")
    export = Path(facts["export"])
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_name") != payload["model_name"]:
        raise PackageError("Model xuất ra không mang đúng tên lượt huấn luyện.")
    log_tail = work / "train_log_tail.txt"
    log_tail.write_bytes(log.read_bytes()[-2_000_000:])
    files = [(f"model/{export.name}", export), ("model/config.json", export_dir / "config.json"),
             ("model/style_vectors.npy", export_dir / "style_vectors.npy"), ("logs/train_log.txt", log_tail)]
    runtime = {"python": pins["train"]["python"], "torch": facts["torch"], "cuda": facts["cuda"], "gpu": gpu_name(),
               "sbv2_commit": pins["train"]["source"]["commit"], "base_assets": base}
    result = write_result(
        output_dir / result_name("train", job["package_id"]), kind="train", package_id=job["package_id"],
        files=files, runtime=runtime,
        payload={"run_id": payload["run_id"], "final_step": facts["step"], "start_step": resumed or 0,
                 "resumed_from_step": resumed or None, "started_at": started_at,
                 "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "losses": facts["losses"][-64:], "weight_evidence": facts["weight"],
                 "model_files": [name for name, _ in files if name.startswith("model/")],
                 "pins_sha256": payload["pins_sha256"], "elapsed_seconds": round(time.time() - started)})
    if drive:
        shutil.copyfile(result, drive / result.name)
        say(f"💾 Đã lưu bản sao kết quả trên Google Drive: {drive / result.name}")
    say(f"✓ Huấn luyện xong: bước {facts['step']}, loss cuối {facts['losses'][-1]}.")
    return result


__all__ = ["build_workspace", "prep_identity", "restore", "run_job", "save_prep"]
