# -*- coding: utf-8 -*-
"""KuroYs AutoMatch — tạo voice trên Google Colab từ gói do tool xuất ra.

Chạy trong notebook KuroYs_Voice_Colab.ipynb (hoặc trên máy có môi trường Voice Engine để test):

    import kuroys_colab_voice as kcv
    result_zip = kcv.run("colab_voice_TenDuAn.zip")

Cách tạo voice giống hệt Voice Engine trong tool (voice_engine_worker.py): chia khối 600 ký tự theo câu,
mỗi khối bỏ tiếng "tách" + quãng im ở mép (voice_cleanup, OmniVoice issue #256), nghỉ 0,08 giây giữa các
khối, hạ âm khi chạm trần, ghi 48 kHz; bản FULL ghép các phần với khoảng nghỉ đã chọn trong tool.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_cleanup  # noqa: E402  (chép nguyên từ tool, xem build_tools/sync_colab_files.py)

PACKAGE_VERSION = 1
RESULT_VERSION = 1
MODEL_RATE = 24000
GENERATION_CHUNK_CHARACTERS = 600
CHUNK_PAUSE_SECONDS = 0.08

Progress = Callable[[str], None]


def _number(value: int) -> str:
    """12345 → "12.345" (kiểu Việt), chỉ đổi dấu trong con số."""
    return f"{int(value):,}".replace(",", ".")


# ------------------------------------------------------------------ chia khối (giống voice_engine_worker)
def _split_oversized_text(value: str, limit: int) -> list:
    parts = []
    remaining = value.strip()
    break_chars = ".!?。！？…;:,\n "
    while len(remaining) > limit:
        cut = limit
        minimum = max(1, int(limit * 0.55))
        best = -1
        for character in break_chars:
            position = remaining.rfind(character, minimum, limit + 1)
            if position > best:
                best = position
        if best >= minimum:
            cut = best + 1
        parts.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def split_generation_text(text: str, limit: int = GENERATION_CHUNK_CHARACTERS) -> list:
    normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not normalized:
        return []
    raw_sentences = re.split(r"(?<=[.!?。！？…])(?:\s+|(?=[^\s]))|\n+", normalized)
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentences.extend(_split_oversized_text(sentence, limit))
    chunks = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------ gói đầu vào
def load_package(package_zip, folder: Path) -> Dict[str, object]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(package_zip) as archive:
            names = set(archive.namelist())
            if "job.json" not in names or "voice.pt" not in names:
                raise ValueError("Gói thiếu job.json hoặc voice.pt. Hãy tạo lại gói bằng nút “Tạo voice bằng Colab”.")
            job = json.loads(archive.read("job.json").decode("utf-8"))
            (folder / "voice.pt").write_bytes(archive.read("voice.pt"))
    except zipfile.BadZipFile:
        raise ValueError("File tải lên không phải gói .zip hợp lệ của KuroYs.") from None
    if int(job.get("kuroys_colab") or 0) != PACKAGE_VERSION:
        raise ValueError("Gói được tạo bằng phiên bản tool khác notebook này. Hãy mở notebook bằng nút trong tool.")
    if not job.get("parts"):
        raise ValueError("Gói không có phần kịch bản nào.")
    job["voice_path"] = str(folder / "voice.pt")
    return job


# ------------------------------------------------------------------ model
def load_model(cache_dir: Optional[str] = None, device: str = "auto", progress: Progress = print):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = str(Path(cache_dir) / "huggingface")
    import torch
    from omnivoice import OmniVoice

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        progress(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        progress("Không có GPU: chạy bằng CPU sẽ RẤT chậm. Nên chọn Runtime → Change runtime type → T4 GPU.")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    progress("Đang tải/nạp model giọng nói (lần đầu khoảng 3 GB, 2–4 phút)...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
    return model, torch, device


# ------------------------------------------------------------------ tạo voice
def generate_part(model, torch, prompt, text: str, *, language: str = "", speed: float = 1.0,
                  steps: int = 16, volume_percent: float = 100.0, progress: Progress = print):
    """Một phần kịch bản → mảng float32 24 kHz đã làm sạch mép và hạ đỉnh."""
    import numpy as np

    chunks = split_generation_text(text)
    if not chunks:
        raise ValueError("Phần kịch bản trống.")
    pause = np.zeros(int(CHUNK_PAUSE_SECONDS * MODEL_RATE), dtype=np.float32)
    arrays = []
    for index, chunk in enumerate(chunks, start=1):
        progress(f"   khối {index}/{len(chunks)}: {chunk[:70]}")
        with torch.inference_mode():
            audio_parts = model.generate(text=chunk, voice_clone_prompt=prompt, language=(language or None),
                                         num_step=int(steps), speed=float(speed))
        if not audio_parts:
            raise RuntimeError(f"Model không trả âm thanh ở khối {index}.")
        for part in audio_parts:
            arrays.append(voice_cleanup.trim_generated_edges(np.asarray(part, dtype=np.float32).reshape(-1),
                                                             MODEL_RATE))
        if index < len(chunks):
            arrays.append(pause)
    audio = np.concatenate(arrays)
    gain = max(0.0, min(150.0, float(volume_percent))) / 100.0
    if gain != 1.0:
        audio = (audio * gain).astype(np.float32, copy=False)
    return voice_cleanup.limit_peak(audio)


def resample(torch, audio, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return audio
    import torchaudio

    tensor = torch.from_numpy(audio).unsqueeze(0)
    return torchaudio.functional.resample(tensor, source_rate, target_rate).squeeze(0).cpu().numpy().astype("float32")


def write_audio(path: Path, audio, rate: int, output_format: str) -> None:
    import soundfile as sf

    path = Path(path)
    temporary = path.with_name(f".{path.stem}.tmp")
    if output_format == "wav":
        sf.write(str(temporary), audio, rate, format="WAV", subtype="PCM_16")
    else:
        try:
            sf.write(str(temporary), audio, rate, format="MP3", bitrate_mode="CONSTANT", compression_level=0.1)
        except TypeError:
            sf.write(str(temporary), audio, rate, format="MP3")
    os.replace(temporary, path)


def merge_parts(arrays: List, rate: int, pause_ms: int):
    """Bản FULL: làm sạch mép từng phần (như bước ghép của tool), chèn khoảng nghỉ, hạ đỉnh."""
    import numpy as np

    silence = np.zeros(int(rate * max(0, int(pause_ms)) / 1000.0), dtype=np.float32)
    pieces = []
    for index, audio in enumerate(arrays):
        pieces.append(voice_cleanup.trim_generated_edges(audio, rate))
        if index < len(arrays) - 1 and silence.size:
            pieces.append(silence)
    return voice_cleanup.limit_peak(np.concatenate(pieces))


def run(package_zip, work_dir: Optional[str] = None, *, cache_dir: Optional[str] = None, device: str = "auto",
        progress: Progress = print, model_bundle=None) -> str:
    """Tạo toàn bộ voice của gói; trả về đường dẫn colab_result_<dự án>.zip."""
    started = time.time()
    work = Path(work_dir or tempfile.mkdtemp(prefix="kuroys_colab_"))
    job = load_package(package_zip, work / "input")
    parts = list(job["parts"])
    total_chars = sum(len(str(part.get("text") or "")) for part in parts)
    progress(f"Gói: dự án “{job['project_name']}”, giọng “{job.get('voice_name')}”, {len(parts)} phần, "
             f"{_number(total_chars)} ký tự.")
    model, torch, device = model_bundle or load_model(cache_dir, device, progress)
    from omnivoice import VoiceClonePrompt

    prompt = VoiceClonePrompt.load(job["voice_path"])
    rate = int(job.get("output_sample_rate") or 48000)
    output_format = "wav" if str(job.get("output_format")) == "wav" else "mp3"
    out = work / "output"
    out.mkdir(parents=True, exist_ok=True)
    finished, arrays = [], []
    for number, part in enumerate(parts, start=1):
        progress(f"Phần {number}/{len(parts)} ({_number(len(str(part['text'])))} ký tự)...")
        part_started = time.time()
        audio = generate_part(model, torch, prompt, str(part["text"]), language=str(job.get("language") or ""),
                              speed=float(job.get("speed") or 1.0), steps=int(job.get("steps") or 16),
                              volume_percent=float(job.get("volume_percent") or 100.0), progress=progress)
        audio = resample(torch, audio, MODEL_RATE, rate)
        write_audio(out / str(part["file"]), audio, rate, output_format)
        arrays.append(audio)
        duration = len(audio) / float(rate)
        finished.append({"index": int(part.get("index") or number), "file": str(part["file"]),
                         "duration_seconds": round(duration, 3), "characters": len(str(part["text"])),
                         "text": str(part["text"]), "source_text": str(part.get("source_text") or part["text"])})
        progress(f"   xong: {duration:.1f} giây voice trong {time.time() - part_started:.0f} giây")
    full = merge_parts(arrays, rate, int(job.get("pause_ms") or 0))
    full_name = str(job.get("full_file") or f"{job['project_name']}_FULL.mp3")
    write_audio(out / full_name, full, rate, "mp3")
    result = {
        "kuroys_colab_result": RESULT_VERSION,
        "package_id": job.get("package_id"),
        "project_name": job.get("project_name"),
        "voice_name": job.get("voice_name"),
        "output_format": output_format,
        "parts": finished,
        "full": {"file": full_name, "duration_seconds": round(len(full) / float(rate), 3)},
        "device": device,
        "elapsed_seconds": round(time.time() - started, 1),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    result_zip = work / f"colab_result_{job['project_name']}.zip"
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("result.json", json.dumps(result, ensure_ascii=False, indent=2))
        for item in finished:
            archive.write(out / item["file"], item["file"])
        archive.write(out / full_name, full_name)
    progress(f"HOÀN TẤT: {len(finished)} phần + bản FULL ({result['full']['duration_seconds']:.1f} giây) "
             f"trong {result['elapsed_seconds']:.0f} giây.")
    return str(result_zip)
