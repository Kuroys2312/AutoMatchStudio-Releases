# -*- coding: utf-8 -*-
"""Entry point used by the notebook: open a package, run it, return the result zip.

Copyright (C) 2026 Kuroys2312.  Licensed under the GNU AGPL v3 (see LICENSE).
"""

from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vj_colab_common import NOTEBOOK_TAG, PackageError, open_job, say  # noqa: E402

JOBS_ROOT = Path("/content/vj_jobs")
OUTPUT_ROOT = Path("/content/vj_out")


def run(zip_path: str | Path, drive_root: str | Path | None = None,
        jobs_root: Path = JOBS_ROOT, output_root: Path = OUTPUT_ROOT) -> Path:
    job, folder = open_job(Path(zip_path), jobs_root)
    payload = job["payload"]
    if job["kind"] == "asr":
        say(f"Gói NHÃN CHỮ (chuyến 1) · mã {job['package_id'][:8]} · {len(payload['audio'])} file âm thanh")
        import vj_colab_asr
        return vj_colab_asr.run_job(job, folder, Path(output_root))
    say(f"Gói HUẤN LUYỆN · mã {job['package_id'][:8]} · giọng \"{payload.get('voice_name', '')}\" · "
        f"{len(payload['train_list'])} đoạn · {payload['train_config']['epochs']} epoch")
    import vj_colab_train
    return vj_colab_train.run_job(job, folder, Path(output_root), Path(drive_root) if drive_root else None)


__all__ = ["NOTEBOOK_TAG", "PackageError", "run"]
