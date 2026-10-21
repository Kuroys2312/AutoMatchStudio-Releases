# -*- coding: utf-8 -*-
"""Trip 1: run the pinned Kotoba Whisper v2.1 stages on uploaded 16-kHz audio.

Returns only the raw provider outputs (raw, stable-ts, punctuated) exactly as
the app's own Kotoba adapter captures them.  Turning them into transcript
segments, matching a script and building the training dataset all happen
back in the app.

Copyright (C) 2026 Kuroys2312.  Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vj_colab_common import (PackageError, gpu_name, load_pins, result_name, say,  # noqa: E402
                             write_result)

# Library identities the app's adapter records as provenance.
_PROVENANCE = {"punctuators": "punctuators", "stable_ts": "stable-ts", "torch": "torch",
               "torchaudio": "torchaudio", "transformers": "transformers"}


def run_job(job: dict, job_dir: Path, output_dir: Path) -> Path:
    from vj_colab_runtime import HF_CACHE, build_venv, provision_kotoba

    payload = job["payload"]
    pins = load_pins()
    python = build_venv("asr", pins)
    snapshot = provision_kotoba(python, pins)
    out = job_dir.parent / "asr_out"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = {**os.environ, "HF_HOME": str(HF_CACHE), "HF_HUB_CACHE": str(HF_CACHE),
           "HF_MODULES_CACHE": str(HF_CACHE / "modules"), "PYTHONIOENCODING": "utf-8",
           "HF_HUB_DISABLE_TELEMETRY": "1"}
    say("Chạy Kotoba Whisper (thô → stable-ts → dấu câu)…")
    completed = subprocess.run([str(python), str(Path(__file__).resolve()), "--worker", str(job_dir / "job.json"),
                                str(snapshot), str(out)], env=env, text=True, capture_output=True)
    if completed.stdout:
        say(completed.stdout[-4000:])
    if completed.returncode:
        raise PackageError("Kotoba thất bại:\n" + completed.stderr[-4000:])
    files = [(f"asr/{item['sha256']}.json", out / f"{item['sha256']}.json") for item in payload["audio"]]
    runtime = json.loads((out / "runtime.json").read_text(encoding="utf-8"))
    runtime["gpu"] = gpu_name()
    return write_result(output_dir / result_name("asr", job["package_id"]), kind="asr",
                        package_id=job["package_id"], files=files, runtime=runtime,
                        payload={"pins_sha256": payload["pins_sha256"], "elapsed_seconds": round(time.time() - started)})


# --------------------------------------------------------------------------
# Worker: runs inside the pinned Python 3.11.9 ``asr`` venv.
# --------------------------------------------------------------------------
def _copy_output(output) -> dict:
    if not isinstance(output, dict) or set(output) != {"text", "chunks"} or type(output["chunks"]) is not list:
        raise RuntimeError("Kotoba runtime returned an unsupported output schema")
    chunks = []
    for item in output["chunks"]:
        if not isinstance(item, dict) or set(item) != {"timestamp", "text"}:
            raise RuntimeError("Kotoba runtime chunk schema is unsupported")
        chunks.append({"timestamp": list(item["timestamp"]), "text": item["text"]})
    return {"text": output.get("text"), "chunks": chunks}


def _jsonable(value):
    """Provider timestamps can be numpy scalars; JSON keeps their exact float value."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return float(value)


def _worker(job_path: Path, snapshot: Path, out: Path) -> None:
    import platform
    from importlib.metadata import version
    from types import MethodType

    pins = load_pins()["asr"]
    if platform.python_version() != pins["python"]:
        raise SystemExit(f"Python {platform.python_version()} != {pins['python']}")
    observed = {name: version(name) for name in pins["library_versions"]}
    if observed != pins["library_versions"]:
        raise SystemExit(f"ASR libraries differ from pins: {observed}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    payload = job["payload"]
    profile = payload["profile"]
    if profile["model_id"] != pins["kotoba"]["model_id"] or profile["revision"] != pins["kotoba"]["revision"]:
        raise SystemExit("package asks for another Kotoba model")

    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if not torch.cuda.is_available():
        raise SystemExit("Không có GPU: vào Runtime → Change runtime type → T4 GPU.")
    dtype = torch.float16 if profile["dtype"] == "float16" else torch.float32
    # The same construction as the app's KotobaAdapter._load.
    pipeline_class = get_class_from_dynamic_module("kotoba_whisper.KotobaWhisperPipeline", str(snapshot),
                                                   revision=profile["revision"], code_revision=profile["revision"],
                                                   local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(snapshot), dtype=dtype, trust_remote_code=True,
                                                      local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True)
    pipe = pipeline_class(model=model, feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer,
                          device=torch.device(profile["device"]), torch_dtype=dtype,
                          punctuator=profile["punctuator"], stable_ts=False)
    original_postprocess = pipe.postprocess

    def compatible_postprocess(_self, model_outputs, *args, **kwargs):
        normalized = []
        for output in model_outputs:
            audio = output.get("audio_array") if isinstance(output, dict) else None
            if isinstance(audio, np.ndarray) and audio.ndim == 1:
                output = dict(output)
                output["audio_array"] = (audio.copy(),)
            normalized.append(output)
        return original_postprocess(normalized, *args, **kwargs)

    pipe.postprocess = MethodType(compatible_postprocess, pipe)
    punctuator_object = getattr(pipe, "punctuator", None)

    def timing_stage(path: Path, stable_ts: bool) -> dict:
        original_stable, original_punctuator = getattr(pipe, "stable_ts", False), getattr(pipe, "punctuator", None)
        try:
            pipe.stable_ts = stable_ts
            pipe.punctuator = None
            return _copy_output(pipe(str(path), return_timestamps=True))
        finally:
            pipe.stable_ts, pipe.punctuator = original_stable, original_punctuator

    provenance = {key: version(name) for key, name in _PROVENANCE.items()}
    for audio in payload["audio"]:
        path = job_path.parent / audio["path"]
        raw = timing_stage(path, False)
        stable = timing_stage(path, True) if profile["stable_ts"] else None
        punctuated = None
        if profile["punctuator"]:
            if punctuator_object is None:
                raise SystemExit("Kotoba punctuator was requested but is unavailable")
            chunks = punctuator_object.punctuate((stable or raw)["chunks"])
            punctuated = {"text": "".join(item["text"] for item in chunks), "chunks": chunks}
        item = {"audio_sha256": audio["sha256"], "profile": profile, "library_versions": provenance,
                "snapshot_manifest_sha256": payload["snapshot_manifest_sha256"],
                "raw_output": _jsonable(raw), "stable_output": _jsonable(stable),
                "punctuated_output": _jsonable(punctuated)}
        (out / f"{audio['sha256']}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        print(f"✓ {audio['sha256'][:12]}…: {len(raw['chunks'])} đoạn", flush=True)
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "library_versions": observed}
    (out / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit("usage: vj_colab_asr.py --worker JOB_JSON SNAPSHOT OUT_DIR")
