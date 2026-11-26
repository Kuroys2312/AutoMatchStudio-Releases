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
# Bản 2: mỗi phần có giọng + cài đặt riêng (gửi cả hàng đợi từ tool).
SUPPORTED_PACKAGE_VERSIONS = (1, 2)
RESULT_VERSION = 1
MODEL_RATE = 24000
GENERATION_CHUNK_CHARACTERS = 600
CHUNK_PAUSE_SECONDS = 0.08
# Không có GPU mà gói dài hơn mức này thì dừng ngay (khách tích "chạy bằng CPU" mới chạy): chạy CPU mất nhiều giờ,
# Colab miễn phí thường ngắt trước khi xong và khách tưởng tool treo.
CPU_MAX_CHARACTERS = 3000
# Ước lượng để báo khách: ~15 ký tự / giây giọng đọc; T4 đo thật 25 s giọng trong 9,6 s (≈ 0,4).
CHARACTERS_PER_VOICE_SECOND = 15.0
T4_SECONDS_PER_VOICE_SECOND = 0.4

Progress = Callable[[str], None]


def _number(value: int) -> str:
    """12345 → "12.345" (kiểu Việt), chỉ đổi dấu trong con số."""
    return f"{int(value):,}".replace(",", ".")


def _duration(seconds: float) -> str:
    """Thời gian dễ đọc: "45 giây", "12 phút", "2 giờ 05 phút"."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} giây"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} phút"
    return f"{minutes // 60} giờ {minutes % 60:02d} phút"


def _package_characters(job: Dict[str, object]) -> int:
    return sum(len(str(part.get("text") or "")) for part in job.get("parts") or [])


def cpu_warning(total_characters: int, has_gpu: bool, allow_cpu: bool = False) -> str:
    """Câu báo dừng khi không có GPU mà gói dài; "" = được chạy."""
    if has_gpu or allow_cpu or int(total_characters) <= CPU_MAX_CHARACTERS:
        return ""
    voice_seconds = int(total_characters) / CHARACTERS_PER_VOICE_SECOND
    return (
        f"❌ Colab CHƯA CÓ GPU nên chưa tạo voice.\n"
        f"Gói này {_number(total_characters)} ký tự ≈ {_duration(voice_seconds)} giọng đọc. Có GPU T4 chỉ mất khoảng "
        f"{_duration(voice_seconds * T4_SECONDS_PER_VOICE_SECOND)}; chạy bằng CPU mất NHIỀU GIỜ và Colab thường tự "
        f"ngắt trước khi xong.\n\n"
        f"Cách bật GPU: menu Thời gian chạy (Runtime) → Thay đổi loại thời gian chạy (Change runtime type) → T4 GPU → "
        f"Lưu (Save), rồi Thời gian chạy → Chạy tất cả (Run all) và chọn lại file .zip ở ô 4.\n"
        f"Nếu Colab báo hết GPU: tài khoản Google này đã dùng hết lượt miễn phí, thử lại sau vài giờ hoặc dùng tài "
        f"khoản Google khác.\n"
        f"Vẫn muốn chạy bằng CPU: tích ô “chay_bang_cpu” ở ô 5 rồi chạy lại ô 5 (nên chia kịch bản thành gói dưới "
        f"{_number(CPU_MAX_CHARACTERS)} ký tự)."
    )


def has_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def check_gpu(package_zip, allow_cpu: bool = False, work_dir: Optional[str] = None) -> str:
    """Gọi ở ô 5 TRƯỚC khi nạp model: "" = chạy được, còn lại là câu báo dừng (không có GPU, gói dài)."""
    if has_gpu() or allow_cpu:
        return ""
    job = load_package(package_zip, Path(work_dir or tempfile.mkdtemp(prefix="kuroys_check_")))
    return cpu_warning(_package_characters(job), False, allow_cpu)


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
            if int(job.get("kuroys_colab") or 0) not in SUPPORTED_PACKAGE_VERSIONS:
                raise ValueError("Gói được tạo bằng phiên bản tool khác notebook này. "
                                 "Hãy mở notebook bằng nút “Mở trang Colab” trong tool.")
            voices = list(job.get("voices") or {"voice.pt": job.get("voice_name")})
            for name in voices:
                if Path(name).name != name or not name.endswith(".pt") or name not in names:
                    raise ValueError(f"Gói thiếu hồ sơ giọng {name}. Hãy tạo lại gói trong tool.")
                (folder / name).write_bytes(archive.read(name))
    except zipfile.BadZipFile:
        raise ValueError("File tải lên không phải gói .zip hợp lệ của KuroYs.") from None
    if not job.get("parts"):
        raise ValueError("Gói không có phần kịch bản nào.")
    job["voice_path"] = str(folder / "voice.pt")
    job["voice_folder"] = str(folder)
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
                  steps: int = 16, volume_percent: float = 100.0, progress: Progress = print,
                  on_chunk: Optional[Callable[[int, int, int, float], str]] = None):
    """Một phần kịch bản → mảng float32 24 kHz đã làm sạch mép và hạ đỉnh.

    ``on_chunk(khối, số khối, số ký tự, giây)`` gọi sau mỗi khối, trả thêm chữ in kèm dòng "xong" (vd. còn bao lâu)."""
    import numpy as np

    chunks = split_generation_text(text)
    if not chunks:
        raise ValueError("Phần kịch bản trống.")
    pause = np.zeros(int(CHUNK_PAUSE_SECONDS * MODEL_RATE), dtype=np.float32)
    arrays = []
    for index, chunk in enumerate(chunks, start=1):
        progress(f"   khối {index}/{len(chunks)}: {chunk[:70]}")
        chunk_started = time.time()
        with torch.inference_mode():
            audio_parts = model.generate(text=chunk, voice_clone_prompt=prompt, language=(language or None),
                                         num_step=int(steps), speed=float(speed))
        if not audio_parts:
            raise RuntimeError(f"Model không trả âm thanh ở khối {index}.")
        seconds = time.time() - chunk_started
        extra = on_chunk(index, len(chunks), len(chunk), seconds) if on_chunk else ""
        progress(f"   ✓ xong khối {index}/{len(chunks)} sau {_duration(seconds)}" + (f" · {extra}" if extra else ""))
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


class NeedGPU(RuntimeError):
    """Không có GPU mà gói dài (xem ``cpu_warning``)."""


def run(package_zip, work_dir: Optional[str] = None, *, cache_dir: Optional[str] = None, device: str = "auto",
        progress: Progress = print, model_bundle=None, allow_cpu: bool = False) -> str:
    """Tạo toàn bộ voice của gói; trả về đường dẫn colab_result_<dự án>.zip.

    Model chạy CPU mà gói dài hơn ``CPU_MAX_CHARACTERS`` → ``NeedGPU`` (trừ khi ``allow_cpu``)."""
    started = time.time()
    work = Path(work_dir or tempfile.mkdtemp(prefix="kuroys_colab_"))
    job = load_package(package_zip, work / "input")
    parts = list(job["parts"])
    total_chars = _package_characters(job)
    progress(f"Gói: dự án “{job['project_name']}”, giọng “{job.get('voice_name')}”, {len(parts)} phần, "
             f"{_number(total_chars)} ký tự.")
    if model_bundle is None:
        warning = cpu_warning(total_chars, device.startswith("cuda") or (device == "auto" and has_gpu()), allow_cpu)
        if warning:
            raise NeedGPU(warning)
    model, torch, device = model_bundle or load_model(cache_dir, device, progress)
    warning = cpu_warning(total_chars, str(device).startswith("cuda"), allow_cpu)
    if warning:
        raise NeedGPU(warning)
    done = {"chars": 0, "seconds": 0.0}

    def on_chunk(index: int, count: int, characters: int, seconds: float) -> str:
        """Tổng tiến độ cả gói + còn khoảng bao lâu (theo tốc độ các khối đã chạy)."""
        done["chars"] += characters
        done["seconds"] += seconds
        percent = 100.0 * done["chars"] / max(1, total_chars)
        left = max(0, total_chars - done["chars"]) * done["seconds"] / max(1, done["chars"])
        return f"cả gói {percent:.0f}%, còn khoảng {_duration(left)}" if left >= 1 else f"cả gói {percent:.0f}%"
    from omnivoice import VoiceClonePrompt

    prompts: Dict[str, object] = {}
    rate = int(job.get("output_sample_rate") or 48000)
    output_format = "wav" if str(job.get("output_format")) == "wav" else "mp3"
    out = work / "output"
    out.mkdir(parents=True, exist_ok=True)
    finished, arrays = [], []
    for number, part in enumerate(parts, start=1):
        progress(f"Phần {number}/{len(parts)} ({_number(len(str(part['text'])))} ký tự)...")
        part_started = time.time()
        # Gói bản 2 (hàng đợi): mỗi phần có giọng + cài đặt riêng; thiếu thì lấy cài đặt chung của gói.
        voice = str(part.get("voice") or "voice.pt")
        if voice not in prompts:
            prompts[voice] = VoiceClonePrompt.load(str(Path(job["voice_folder"]) / voice))
        part_format = "wav" if str(part.get("output_format") or output_format) == "wav" else "mp3"
        audio = generate_part(model, torch, prompts[voice], str(part["text"]), language=str(job.get("language") or ""),
                              speed=float(part.get("speed") or job.get("speed") or 1.0),
                              steps=int(part.get("steps") or job.get("steps") or 16),
                              volume_percent=float(part.get("volume_percent") or job.get("volume_percent") or 100.0),
                              progress=progress, on_chunk=on_chunk)
        audio = resample(torch, audio, MODEL_RATE, rate)
        write_audio(out / str(part["file"]), audio, rate, part_format)
        arrays.append(audio)
        duration = len(audio) / float(rate)
        finished.append({"index": int(part.get("index") or number), "file": str(part["file"]),
                         "duration_seconds": round(duration, 3), "characters": len(str(part["text"])),
                         "text": str(part["text"]), "source_text": str(part.get("source_text") or part["text"]),
                         "queue_item_id": str(part.get("queue_item_id") or "")})
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
