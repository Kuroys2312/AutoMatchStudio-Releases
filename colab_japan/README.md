# KuroYs Voice Japan: notebook Google Colab

Notebook và mã chạy trên Colab để **huấn luyện (clone) giọng tiếng Nhật** cho app *KuroYs Voice Japan* khi máy không đủ GPU. Việc tạo giọng (TTS) và phụ đề vẫn chạy trên máy.

- Notebook: `KuroYs_VoiceJapan_Colab.ipynb`. App mở đúng bản qua link có ghim tag, ví dụ `vj-colab-v1`.
- Tag đã phát hành thì không bao giờ dời. Khi notebook thay đổi sẽ ra tag mới (`vj-colab-v2`, …).

## Cách hoạt động

1. App đóng gói một file `.zip`, gồm `job.json` và các file cần thiết. `job.json` liệt kê từng file kèm kích thước và SHA-256.
2. Người dùng mở notebook, bấm **Run all**, rồi tải gói lên. Notebook kiểm tra định dạng gói, mã kiểm tra từng file và bộ phiên bản đã ghim.
3. Notebook dựng môi trường riêng bằng `uv`: Python 3.11.9 cùng đúng các phiên bản thư viện trong `pins.json`. Model được tải từ Hugging Face theo revision cố định và kiểm SHA-256.
4. Có hai loại gói:
   - Gói `asr`: chạy Kotoba Whisper v2.1 (thô, stable-ts, dấu câu) và trả về đầu ra thô.
   - Gói `train`: chạy nguyên bản `preprocess_text.py`, `bert_gen.py`, `style_gen.py` và `train_ms_jp_extra.py` của Style-Bert-VITS2 2.5.0 (commit `250e83e9`). Trả về model suy luận cùng loss và bằng chứng trọng số đã đổi.
5. Notebook ghi `result.json` liệt kê từng file kèm SHA-256. App kiểm tra lại toàn bộ trước khi nhập.

Không web server, không tunnel, không proxy. Không cần token. Notebook không ghi khoá hay đường dẫn cá nhân nào vào gói.

## File

| File | Vai trò |
|---|---|
| `vj_colab_common.py` | Định dạng gói, kiểm tra SHA-256, ghi kết quả (chỉ dùng thư viện chuẩn) |
| `vj_colab_runtime.py` | Dựng venv bằng uv, lấy mã nguồn SBV2 theo commit, tải và kiểm model |
| `vj_colab_asr.py` | Chuyến 1: nhãn chữ bằng Kotoba Whisper |
| `vj_colab_train.py` | Chuyến 2: huấn luyện JP-Extra, sao lưu checkpoint lên Drive, đóng gói model |
| `vj_colab_main.py` | Điểm vào cho notebook |
| `pins.json` | Bản chép phiên bản ghim từ app. Sinh bằng `scripts/sync_colab_japan.py --write-pins` |

## Giấy phép

Mã trong thư mục này phát hành theo **GNU AGPL v3** (xem `LICENSE`), cùng giấy phép với Style-Bert-VITS2 mà nó chạy. Các model được tải từ nguồn gốc trong phiên Colab của người dùng, không được chép vào đây:

| Model | Giấy phép |
|---|---|
| JP-Extra base | Theo model card của litagin |
| BERT ku-nlp | CC BY-SA 4.0 |
| WavLM | Theo model card của Microsoft |
| WeSpeaker | CC BY 4.0 |
| Kotoba Whisper | Apache-2.0 |

Chỉ huấn luyện giọng của chính bạn hoặc giọng đã được chủ sở hữu cho phép. Không dùng để mạo danh hay gây hại.
