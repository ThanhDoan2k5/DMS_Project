import cv2 
import mediapipe as mp 
import time
import pygame 
import torch
from ultralytics import YOLO 
import numpy as np 
import csv

from sleep_detector import SleepDetector
from smoking_detector import SmokingDetector
from distraction_detector import DistractionDetector

# ================= 1. KHỞI TẠO HỆ THỐNG VÀ AI MODELS =================
pygame.mixer.init()
music_path = "alert_music.mp3" 
music_playing = False 

device_to_use = 0 if torch.cuda.is_available() else 'cpu'
if device_to_use == 0:
    print("🔥 ĐÃ KÍCH HOẠT CARD ĐỒ HỌA RỜI (GPU) CHO YOLO!")
    torch.cuda.empty_cache() 
else:
    print("⚠️ YOLO đang chạy bằng CPU.")

try:
    model_smoke = YOLO('best.pt') 
    model_phone = YOLO('best_phone.pt') 
    print("✅ Đã nạp thành công 2 mô hình AI Gốc (Smoke & Phone)!")
except Exception as e:
    model_smoke, model_phone = None, None
    print(f"❌ Lỗi nạp mô hình AI: {e}")

# Trả về 1 mặt người vì khoang lái chỉ focus vào tài xế
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

sleep_detector = SleepDetector()
smoking_detector = SmokingDetector()
distraction_detector = DistractionDetector()

last_time = time.time()
prev_frame_time = 0
frame_counter = 0

# Bật lại tính năng nhảy khung hình để tối ưu FPS (Chỉ chạy YOLO 1 lần / 3 frame)
YOLO_SKIP_FRAMES = 3  
cached_obj_boxes = {'smoke': [], 'phone': [], 'fake_smoke': []}
measured_latencies = []

csv_file = open('dms_real_test.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'ground_truth', 'prediction'])
current_ground_truth = 0  

# ================= 3. VÒNG LẶP XỬ LÝ CHÍNH (CHẾ ĐỘ WEBCAM) =================
cap = cv2.VideoCapture(0) # MỞ WEBCAM GỐC
print("🚀 HỆ THỐNG DMS ĐÃ KHỞI ĐỘNG (CHẾ ĐỘ WEBCAM DEMO)!")
print("--- HƯỚNG DẪN GÁN NHÃN THỰC TẾ ---")
print("[0]: Bình thường  |  [1]: Ngủ gật  |  [2]: Phân tâm  |  [3]: Hút thuốc")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Bật lại LẬT GƯƠNG cho Webcam
    frame = cv2.flip(frame, 1) 
    
    start_measure_time = time.perf_counter()
    
    h_frame, w_frame, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    frame_counter += 1

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 30
    prev_frame_time = new_frame_time

    is_calling_now = False
    obj_boxes = {'smoke': [], 'phone': [], 'fake_smoke': []}

    if model_smoke and model_phone:
        if frame_counter % YOLO_SKIP_FRAMES == 0:
            obj_boxes = {'smoke': [], 'phone': [], 'fake_smoke': []}
            
            # --- MODEL 1: QUÉT THUỐC LÁ (KHẮT KHE) ---
            # Trả về phân giải 320px và conf=0.75 để chạy mượt và loại bỏ nhiễu
            res_smoke = model_smoke(frame, conf=0.75, iou=0.40, max_det=3, imgsz=320, device=device_to_use, verbose=False)
            for r in res_smoke:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1
                    if w == 0 or h == 0: continue
                    aspect_ratio = max(w, h) / min(w, h)
                    
                    # BẬT LẠI BỘ LỌC HÌNH HỌC (Chặn bút bi, đũa ăn, vật thể dài)
                    if aspect_ratio >= 3.5: 
                        obj_boxes['fake_smoke'].append((x1, y1, x2, y2, aspect_ratio))
                        continue
                    elif aspect_ratio < 1.7 and conf < 0.75: 
                        continue
                    
                    obj_boxes['smoke'].append((x1, y1, x2, y2, conf))
            
            # --- MODEL 2: QUÉT ĐIỆN THOẠI ---
            res_phone = model_phone(frame, conf=0.80, max_det=2, imgsz=320, device=device_to_use, verbose=False)
            for r in res_phone:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    obj_boxes['phone'].append((x1, y1, x2, y2, conf))
            
            cached_obj_boxes = obj_boxes
        else:
            obj_boxes = cached_obj_boxes

        for box in obj_boxes['smoke']:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Smoke {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for box in obj_boxes['fake_smoke']:
            x1, y1, x2, y2, ar = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2) 
            cv2.putText(frame, f"Not Smoke (AR:{ar:.1f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        for box in obj_boxes['phone']:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) 
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            is_calling_now = True

    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            sleep_detector.process(landmarks, dt)
            distraction_detector.process(frame, landmarks, is_calling_now, dt)
            smoking_detector.process(frame, landmarks, obj_boxes, dt)
    else:
        distraction_detector.process(frame, None, is_calling_now, dt)

    music_playing = sleep_detector.draw_alert(frame, music_playing, music_path, pygame)
    smoking_detector.draw_alert(frame)
    distraction_detector.draw_alert(frame)

    latency_ms = (time.perf_counter() - start_measure_time) * 1000
    measured_latencies.append(latency_ms)

    fps_color = (0, 255, 0) if fps > 25 else ((0, 165, 255) if fps > 15 else (0, 0, 255))
    cv2.putText(frame, f"FPS: {int(fps)}", (w_frame - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 3)
    cv2.putText(frame, f"Lat: {latency_ms:.1f}ms", (w_frame - 160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    status_dict = {0: "Normal", 1: "Drowsiness", 2: "Distraction", 3: "Smoking"}
    cv2.putText(frame, f"GT Mode: [{current_ground_truth}] {status_dict[current_ground_truth]}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    log_text = f"Slp: {sleep_detector.timer_sleep:.1f}s | Smk: {smoking_detector.timer_smoking:.1f}s | Dist: {distraction_detector.timer_distracted:.1f}s"
    cv2.putText(frame, log_text, (20, h_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('DMS Ultimate Final', frame)

    ai_prediction = 0
    if sleep_detector.timer_sleep >= 1.5:
        ai_prediction = 1
    elif distraction_detector.timer_distracted >= 2.0:
        ai_prediction = 2
    elif smoking_detector.timer_smoking >= 1.0: # Trả về chuẩn 1.0 giây
        ai_prediction = 3

    csv_writer.writerow([time.time(), current_ground_truth, ai_prediction])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('0'): 
        current_ground_truth = 0
    elif key == ord('1'): 
        current_ground_truth = 1
    elif key == ord('2'): 
        current_ground_truth = 2
    elif key == ord('3'): 
        current_ground_truth = 3

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
csv_file.close()