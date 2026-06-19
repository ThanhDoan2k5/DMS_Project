import cv2
import math
import numpy as np
from utils import euclidean_distance

class DistractionDetector:
    def __init__(self):
        self.GAZE_LEFT_THRESH = 0.75  
        self.GAZE_RIGHT_THRESH = 1.25 
        self.timer_distracted = 0.0
        
        # Mô hình 3D Head Pose
        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0], [0.0, 330.0, -65.0], [-225.0, -170.0, -135.0], 
            [225.0, -170.0, -135.0], [-150.0, 150.0, -125.0], [150.0, 150.0, -125.0]
        ], dtype=np.float64)

    def process(self, frame, landmarks, is_calling_now, dt):
        h_frame, w_frame, _ = frame.shape
        is_looking_away_now = False

        if landmarks is not None:
            # Đảo mắt lươn lẹo
            D1_L = euclidean_distance(landmarks.landmark[33], landmarks.landmark[468])
            D2_L = euclidean_distance(landmarks.landmark[468], landmarks.landmark[133])
            gaze_left = D1_L / D2_L if D2_L > 0 else 1.0

            D1_R = euclidean_distance(landmarks.landmark[263], landmarks.landmark[473])
            D2_R = euclidean_distance(landmarks.landmark[473], landmarks.landmark[362])
            gaze_right = D1_R / D2_R if D2_R > 0 else 1.0

            gaze_ratio = (gaze_left + gaze_right) / 2.0
            if gaze_ratio < self.GAZE_LEFT_THRESH or gaze_ratio > self.GAZE_RIGHT_THRESH:
                is_looking_away_now = True

            # Tư thế đầu (Head Pose)
            face_2d = np.array([[int(landmarks.landmark[idx].x * w_frame), int(landmarks.landmark[idx].y * h_frame)] 
                                for idx in [1, 152, 33, 263, 61, 291]], dtype=np.float64)
            cam_matrix = np.array([[w_frame, 0, w_frame / 2], [0, w_frame, h_frame / 2], [0, 0, 1]])
            success, rvec, tvec = cv2.solvePnP(self.face_3d_model, face_2d, cam_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
            
            if success:
                R, _ = cv2.Rodrigues(rvec)
                pitch = math.degrees(math.asin(R[2, 1]))
                yaw = math.degrees(math.atan2(-R[2, 0], R[2, 2]))
                if abs(yaw) > 30.0 or pitch < -30.0 or pitch > 30.0: 
                    is_looking_away_now = True
                
                cv2.putText(frame, f"Pitch: {int(pitch)} | Yaw: {int(yaw)} | Gaze: {gaze_ratio:.2f}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Nếu mất landmark mặt hoàn toàn nhưng vẫn đang cầm máy gọi
            if is_calling_now:
                is_looking_away_now = True

        if is_calling_now or is_looking_away_now: 
            self.timer_distracted += dt
        else: 
            # Giảm tốc độ nhanh hơn (X8) như yêu cầu tối ưu hóa phản ứng
            self.timer_distracted = max(0.0, self.timer_distracted - dt * 8.0)
            
        return self.timer_distracted

    def draw_alert(self, frame):
        if self.timer_distracted > 4.0:
            cv2.putText(frame, "BAO DONG: MAT TAP TRUNG!", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif self.timer_distracted > 2.0:
            cv2.putText(frame, "NHAC NHO: VUI LONG NHIN DUONG", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
