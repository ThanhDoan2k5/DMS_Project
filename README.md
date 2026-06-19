Markdown
# 🚘 Hệ Thống Giám Sát Người Lái (Driver Monitoring System - DMS)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-blueviolet.svg)

Đây là kho lưu trữ mã nguồn chính thức cho Đồ án Kỹ thuật Điện tử và Tin học với đề tài **"Xây dựng hệ thống giám sát hành vi người lái xe dựa trên kỹ thuật Thị giác máy tính và Học sâu"**.

**Trường Đại học Khoa học Tự nhiên - Đại học Quốc gia Hà Nội (HUS)** **Sinh viên thực hiện:** Đoàn Quang Thành (Mã SV: 23001628)  
**Giảng viên hướng dẫn:** CN. Vi Anh Quân  

---

## 🌟 Các tính năng cốt lõi (Core Features)

Hệ thống hoạt động theo thời gian thực (Real-time) qua Webcam, bao gồm 3 phân hệ chính:
1. **💤 Giám sát Ngủ gật (Sleep Detection):** - Sử dụng chỉ số EAR (Eye Aspect Ratio).
   - Tích hợp giải thuật **Tự động hiệu chỉnh động (Dynamic Calibration)** trong 3 giây đầu để thích ứng với cấu trúc mắt của từng tài xế.
2. **📵 Cảnh báo Mất tập trung (Distraction Detection):** - Phân tích tư thế đầu (Head Pose - Góc Yaw, Pitch) qua thuật toán `SolvePnP`.
   - Theo dõi ánh nhìn tròng mắt (Gaze Tracking).
   - Nhận diện hành vi cầm điện thoại áp tai (YOLOv8).
3. **🚬 Phát hiện Hút thuốc (Smoking Detection):**
   - Đánh giá giao thoa không gian giữa vật thể và vùng miệng (IOMR - Intersection Over Mouth Ratio).
   - Tích hợp **Bộ lọc Tỷ lệ khía cạnh (Aspect Ratio Filter)** để triệt tiêu báo động giả do các vật dụng sinh hoạt (bút bi, ống hút).

---

## 📁 Cấu trúc thư mục dự án

```text
DMS_Project/
│
├── main.py                   # Luồng điều phối trung tâm (Main Pipeline)
├── utils.py                  # Các hàm toán học tính toán EAR, IOMR, v.v.
├── sleep_detector.py         # Lớp đối tượng xử lý hành vi nhắm mắt
├── smoking_detector.py       # Lớp đối tượng phân tích hành vi hút thuốc
├── distraction_detector.py   # Lớp đối tượng giải mã tư thế đầu và ánh nhìn
├── eval_metrics.py           # Script đánh giá và vẽ Ma trận nhầm lẫn
├── requirements.txt          # Danh sách các thư viện phụ thuộc
├── best.pt                   # [Cần tải thêm] Trọng số YOLO nhận diện thuốc lá
├── best_phone.pt             # [Cần tải thêm] Trọng số YOLO nhận diện điện thoại
└── alert_music.mp3           # [Cần tải thêm] Âm thanh cảnh báo khẩn cấp
⚙️ Hướng dẫn Cài đặt & Chạy thử (Quick Start)
Bước 1: Clone kho lưu trữ về máy tính
Bash
git clone [https://github.com/TenCuaOng/DMS_Project.git](https://github.com/TenCuaOng/DMS_Project.git)
cd DMS_Project
Bước 2: Tạo môi trường ảo và cài đặt thư viện
Khuyến nghị sử dụng Anaconda với Python 3.9 để tương thích tốt nhất với các thư viện AI:

Bash
conda create -n dms_env python=3.9 -y
conda activate dms_env
pip install -r requirements.txt
(Lưu ý: Bạn cần đặt 2 file trọng số best.pt, best_phone.pt và file âm thanh alert_music.mp3 vào thư mục gốc của dự án trước khi chạy).

Bước 3: Khởi chạy hệ thống thời gian thực
Bash
python main.py
Các phím tắt điều khiển trong quá trình chạy:

Phím Q: Tắt camera và kết thúc chương trình.

Phím 0, 1, 2, 3: Thay đổi nhãn thực tế (Ground Truth) để ghi Log vào file CSV phục vụ công tác đánh giá.

📊 Đánh giá Hiệu năng (Evaluation)
Hệ thống cung cấp một script để tự động xuất Báo cáo phân loại (Classification Report) và vẽ Ma trận nhầm lẫn (Confusion Matrix) dựa trên dữ liệu thu thập được trong quá trình chạy thực tế.

Để xem kết quả đánh giá, chạy lệnh sau:

Bash
python eval_metrics.py
Kết quả sẽ xuất ra màn hình Terminal và lưu biểu đồ vào file Confusion_Matrix_Real.png.

Bản quyền thuộc về tác giả. Mã nguồn được công khai phục vụ mục đích học thuật và đánh giá của Hội đồng nghiệm thu Khoa Vật lý - HUS.
