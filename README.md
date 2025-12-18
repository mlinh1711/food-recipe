# 🍲 Bếp Việt nè – Food2Recipe AI

**Bếp Việt nè** là hệ thống AI nhận diện món ăn Việt Nam từ hình ảnh và gợi ý công thức nấu ăn chuẩn vị. Dự án không chỉ dừng lại ở việc nhận diện (classification) mà còn xây dựng một **Recommender System** thông minh, gợi ý các món tương tự và học hỏi từ phản hồi của người dùng trong phiên làm việc.

---

## ✨ Tính năng nổi bật

*   **⚡️ Nhận diện món ăn (AI Recognition):**
    *   Sử dụng mô hình Vision Transformer (ViT-B-32) để trích xuất đặc trưng ảnh.
    *   Hệ thống tìm kiếm (Retrieval) dựa trên FAISS index để so khớp ảnh tải lên với kho dữ liệu 30 món ăn Việt Nam phổ biến.

*   **📖 Công thức nấu chi tiết:**
    *   Hiển thị Tên tiếng Việt, Nguyên liệu và Cách nấu từng bước cho món được nhận diện.

*   **🔄 Gợi ý thông minh (Recommender System):**
    *   **Món tương tự (Visual Similarity):** Gợi ý các món có hình ảnh/đặc điểm gần giống.
    *   **Khám phá nhóm (Group Exploration):** Gợi ý các món cùng loại (ví dụ: các loại Bún, Bánh, Chè...).

*   **👤 Cá nhân hóa theo phiên (Session-based Feedback):**
    *   **Feedback Loop:** Người dùng có thể bấm **Chính xác/Sai rồi** hoặc **Thích/Không thích**.
    *   **Real-time Reranking:** Hệ thống lập tức cập nhật thứ tự gợi ý dựa trên lịch sử tương tác của bạn trong phiên hiện tại.
    *   **Giao diện 2 thẻ:** Tách biệt rõ ràng giữa "Kết quả nhận diện gốc" và "Món bạn đang xem/khám phá".

---

## 📂 Cấu trúc dự án

```text
food-recipe/
├── data/
│   ├── Images/               # Chứa ảnh dataset (Train/Validate/Test)
│   └── vnfood30_recipes.csv  # Dữ liệu công thức & tên tiếng Việt
├── food2recipe/
│   ├── app/                  # Mã nguồn Streamlit UI & UI components
│   ├── core/                 # Cấu hình hệ thống (Settings)
│   ├── preprocessing/        # Xử lý ảnh & text
│   ├── retrieval/            # Logic AI: Recommender, Search Engine, Related Items
│   └── scripts/              # Các script build index
├── tools/
│   └── build_centroids.py    # Script tạo centroid cho recommender
├── .env.example              # Mẫu cấu hình môi trường
├── requirements.txt          # Các thư viện cần thiết
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Chuẩn bị môi trường

Yêu cầu: Python 3.8+.

```bash
# 1. Tạo môi trường ảo (khuyên dùng)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình (`.env`)

Tạo file `.env` từ file mẫu và kiểm tra đường dẫn dữ liệu:

```bash
cp .env.example .env
```

*Lưu ý: Mặc định hệ thống tìm dữ liệu trong thư mục `data/`.*

### 3. Build Artifacts (Bắt buộc)

Trước khi chạy app, bạn cần tạo Index và Centroids cho hệ thống AI.

**Bước 1: Build Image Index** (Quét ảnh và tạo vector search)
```bash
python -m food2recipe.scripts.build_index
```

**Bước 2: Build Centroids** (Tạo dữ liệu cho tính năng gợi ý)
```bash
python -m tools.build_centroids
```

### 4. Khởi chạy Ứng dụng

```bash
python -m streamlit run food2recipe/app/streamlit_app.py
```

Ứng dụng sẽ chạy tại: `http://localhost:8501`

---

## 📱 Hướng dẫn sử dụng

1.  **Upload ảnh:** Kéo thả ảnh món ăn vào khung upload.
2.  **Xem kết quả:**
    *   Thẻ trên cùng hiển thị **Kết quả nhận diện gốc** (AI dự đoán).
    *   Thẻ dưới hiển thị **Bạn đang xem** (Món hiện tại + Công thức).
3.  **Tương tác:**
    *   Bấm **✅ Chính xác** hoặc **❌ Sai rồi** để sửa kết quả.
    *   Bấm **👍 Thích / 👎 Không** để "dạy" hệ thống gu ăn uống của bạn.
4.  **Khám phá:**
    *   Click vào các món ở mục **"Món tương tự"** hoặc **"Khám phá thêm"** bên dưới.
    *   Giao diện sẽ chuyển sang món mới nhưng vẫn giữ lại kết quả nhận diện gốc để bạn đối chiếu.

---

## 🛠 Troubleshooting

*   **Lỗi `FileNotFoundError` khi build:**
    Kiểm tra lại xem bạn đã giải nén dataset vào đúng thư mục `data/Images` chưa. Cấu trúc đúng là `data/Images/Train/...`.
*   **App báo "Hệ thống chưa sẵn sàng":**
    Bạn chưa chạy bước 3 (Build Artifacts). Hãy chạy `build_index` và `build_centroids`.
*   **Lỗi `DuplicateWidgetID`:**
    Đã được fix trong phiên bản mới nhất, đảm bảo bạn đang dùng code mới nhất từ repo.

---
**Credits:** Dự án sử dụng mô hình pre-trained OpenCLIP và dataset VnFood30.
