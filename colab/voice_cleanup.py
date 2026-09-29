# -*- coding: utf-8 -*-
"""Làm sạch mép âm thanh cho Voice Studio (chỉ dùng numpy).

Vì sao cần: OmniVoice nhân bản giọng theo kiểu "đọc tiếp" giọng mẫu. Giọng mẫu bị
cắt cụt ở cuối (không có khoảng lặng) thì model đọc nốt phần dở trước, sinh ra
một tiếng "tách" ngắn ở đầu mỗi đoạn voice, rồi một quãng im 0,5 giây mới tới lời
(OmniVoice issue #256). Bộ cắt khoảng lặng của OmniVoice dùng ngưỡng -50 dB nên
dừng ngay ở tiếng tách (~-42 dB) và giữ nguyên quãng im phía sau.

Hai lớp sửa:
* ``trim_generated_edges``: lưới an toàn cho MỌI voice tạo ra (kể cả hồ sơ giọng
  cũ): bỏ tiếng lẻ ngắn ở đầu/cuối, chỉ giữ một chút trước chữ đầu và sau chữ cuối.
* ``settle_reference``: sửa gốc khi chuẩn bị giọng mẫu: không cắt giữa chữ, thêm
  khoảng lặng thật ở cuối để model không phải "đọc nốt".
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

FRAME_SECONDS = 0.01
# Ngưỡng "có tiếng": thấp hơn mức lời nói to (phân vị 90) chừng này dB, không thấp hơn SILENCE_FLOOR_DB.
SPEECH_BELOW_PEAK_DB = 32.0
SILENCE_FLOOR_DB = -50.0
# Tiếng lẻ ngắn hơn chừng này, cách lời thật một quãng im ít nhất BLIP_GAP, là tiếng tách/thở thừa.
BLIP_MAX_SECONDS = 0.15
BLIP_GAP_SECONDS = 0.2
PEAK_CEILING = 0.98


def _frame_db(samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    hop = max(1, int(round(sample_rate * FRAME_SECONDS)))
    count = int(np.ceil(len(samples) / hop))
    padded = np.zeros(count * hop, dtype=np.float32)
    padded[: len(samples)] = samples
    rms = np.sqrt(np.mean(padded.reshape(count, hop) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(np.maximum(rms, 1e-6)), hop


def _threshold(db: np.ndarray) -> float:
    return max(SILENCE_FLOOR_DB, float(np.percentile(db, 90)) - SPEECH_BELOW_PEAK_DB)


def _islands(loud: np.ndarray) -> List[Tuple[int, int]]:
    """Các đoạn khung liên tiếp có tiếng: [(đầu, cuối+1)]."""
    runs: List[Tuple[int, int]] = []
    start = None
    for index, value in enumerate(loud):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(loud)))
    return runs


def _drop_edge_blips(runs: List[Tuple[int, int]], blip_frames: int, gap_frames: int) -> List[Tuple[int, int]]:
    runs = list(runs)
    while len(runs) > 1 and runs[0][1] - runs[0][0] <= blip_frames and runs[1][0] - runs[0][1] >= gap_frames:
        runs.pop(0)
    while len(runs) > 1 and runs[-1][1] - runs[-1][0] <= blip_frames and runs[-1][0] - runs[-2][1] >= gap_frames:
        runs.pop()
    return runs


def _fade(samples: np.ndarray, sample_rate: int, fade_in: float, fade_out: float) -> np.ndarray:
    out = samples.astype(np.float32, copy=True)
    size_in = min(len(out) // 2, int(sample_rate * fade_in))
    if size_in > 0:
        out[:size_in] *= np.linspace(0.0, 1.0, size_in, dtype=np.float32)
    size_out = min(len(out) // 2, int(sample_rate * fade_out))
    if size_out > 0:
        out[-size_out:] *= np.linspace(1.0, 0.0, size_out, dtype=np.float32)
    return out


def speech_bounds(samples, sample_rate: int) -> Tuple[int, int]:
    """(mẫu đầu, mẫu cuối+1) của phần lời thật, đã bỏ tiếng lẻ ở hai mép. Không có lời thì (0, len)."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0, 0
    db, hop = _frame_db(audio, sample_rate)
    runs = _islands(db > _threshold(db))
    if not runs:
        return 0, audio.size
    runs = _drop_edge_blips(runs, int(round(BLIP_MAX_SECONDS / FRAME_SECONDS)),
                            int(round(BLIP_GAP_SECONDS / FRAME_SECONDS)))
    return runs[0][0] * hop, min(audio.size, runs[-1][1] * hop)


def trim_generated_edges(samples, sample_rate: int, *, lead: float = 0.06, tail: float = 0.15,
                         fade_in: float = 0.015, fade_out: float = 0.02) -> np.ndarray:
    """Bỏ tiếng tách + quãng im ở đầu/cuối một đoạn voice vừa tạo; không bao giờ cắt vào lời."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    first, last = speech_bounds(audio, sample_rate)
    if last <= first:
        return audio
    start = max(0, first - int(sample_rate * lead))
    end = min(audio.size, last + int(sample_rate * tail))
    trimmed = audio[start:end]
    # Chỉ làm mờ phần đệm vừa giữ lại (trước chữ đầu / sau chữ cuối), không chạm vào lời.
    fade_in = min(fade_in, (first - start) / sample_rate) if first > start else 0.0
    fade_out = min(fade_out, (end - last) / sample_rate) if end > last else 0.0
    return _fade(trimmed, sample_rate, fade_in, fade_out)


def limit_peak(samples, ceiling: float = PEAK_CEILING) -> np.ndarray:
    """Hạ cả đoạn cho đỉnh không chạm trần (thay cho cắt cụt đỉnh gây rè)."""
    audio = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > ceiling:
        audio = (audio * (ceiling / peak)).astype(np.float32, copy=False)
    return audio


def _find_pause(loud: np.ndarray, lo: int, hi: int, min_frames: int, *, last: bool) -> Tuple[int, int]:
    """Quãng im dài nhất-gần-mép (ít nhất min_frames khung) trong [lo, hi). Trả (-1, -1) nếu không có."""
    best = (-1, -1)
    start = None
    for index in range(max(0, lo), min(len(loud), hi) + 1):
        quiet = index < min(len(loud), hi) and not loud[index]
        if quiet and start is None:
            start = index
        elif not quiet and start is not None:
            if index - start >= min_frames:
                best = (start, index)
                if not last:
                    return best
            start = None
    return best


def settle_reference(samples, sample_rate: int, *, allow_shift: bool = True, head_pad: float = 0.05,
                     tail_pad: float = 0.35, max_shift_end: float = 1.5, max_shift_start: float = 0.8,
                     min_seconds: float = 3.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Giọng mẫu sạch mép cho OmniVoice: bắt đầu/kết thúc ở chỗ nghỉ, có khoảng lặng thật ở cuối.

    allow_shift=False khi lời mẫu đã chốt theo đúng đoạn cắt (file 09): chỉ bỏ quãng im thừa, làm
    mờ mép và thêm khoảng lặng, không dời điểm cắt (dời thì lời mẫu không còn khớp).
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    info = {"cut_start": 0.0, "cut_end": 0.0, "tail_pad": tail_pad}
    if audio.size == 0:
        return audio, info
    db, hop = _frame_db(audio, sample_rate)
    loud = db > _threshold(db)
    first, last = speech_bounds(audio, sample_rate)
    start, end = first, last
    min_samples = int(min_seconds * sample_rate)
    pause_frames = int(round(0.12 / FRAME_SECONDS))
    if allow_shift:
        # Kết thúc giữa chữ (mép cuối còn tiếng, không có quãng im sau): lùi về chỗ nghỉ gần nhất.
        if last >= audio.size - hop * 2:
            lo = int((audio.size - max_shift_end * sample_rate) / hop)
            pause = _find_pause(loud, lo, len(loud) - 2, pause_frames, last=True)
            if pause[0] >= 0 and pause[0] * hop - start >= min_samples:
                end = pause[0] * hop + int(0.08 * sample_rate)
        # Bắt đầu giữa chữ: tiến tới chỗ nghỉ đầu tiên.
        if first <= hop * 2:
            hi = int(max_shift_start * sample_rate / hop)
            pause = _find_pause(loud, 2, hi, pause_frames, last=False)
            if pause[1] >= 0 and end - pause[1] * hop >= min_samples:
                start = max(0, pause[1] * hop - int(0.05 * sample_rate))
    lead_keep = int(0.05 * sample_rate)
    tail_keep = int(0.1 * sample_rate)
    begin = max(0, start - lead_keep)
    finish = min(audio.size, end + tail_keep)
    info["cut_start"] = round((begin - max(0, first - lead_keep)) / sample_rate, 3)
    info["cut_end"] = round((min(audio.size, last + tail_keep) - finish) / sample_rate, 3)
    body = _fade(audio[begin:finish], sample_rate, 0.01, 0.03)
    padded = np.concatenate([
        np.zeros(int(head_pad * sample_rate), dtype=np.float32), body,
        np.zeros(int(tail_pad * sample_rate), dtype=np.float32),
    ])
    return padded, info
