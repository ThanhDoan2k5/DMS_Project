import cv2
from collections import deque
from utils import LEFT_EYE, RIGHT_EYE, LEFT_BROW, LEFT_TOP, LEFT_BOTTOM, RIGHT_BROW, RIGHT_TOP, RIGHT_BOTTOM, calculate_ear, calculate_r_ratio

class SleepDetector:
    def __init__(self):
        # TỰ ĐỘNG HIỆU CHỈNH (DYNAMIC CALIBRATION)
        self.is_calibrated = False
        self.calibration_frames = 0
        self.max_cal_frames = 60 # Thu thập dữ liệu trong khoảng 2-3 giây đầu tiên
        self.ear_sum = 0.0
        self.baseline_ear = 0.0

        # Ngưỡng đóng mắt sẽ được AI tự tính toán lại, không gắn cứng nữa
        self.EAR_CLOSED_THRESH = 0.20 
        
        self.t_blink = 0.08
        self.r_buffer = deque(maxlen=3)
        self.timer_sleep = 0.0

    def process(self, landmarks, dt):
        ear = (calculate_ear(LEFT_EYE, landmarks) + calculate_ear(RIGHT_EYE, landmarks)) / 2.0
        
        # BƯỚC 1: Quá trình học cấu tạo mắt trong 3 giây đầu
        if not self.is_calibrated:
            self.ear_sum += ear
            self.calibration_frames += 1
            if self.calibration_frames >= self.max_cal_frames:
                self.baseline_ear = self.ear_sum / self.max_cal_frames
                # Chốt ngưỡng nhắm mắt = 75% độ mở mắt bình thường của tài xế này
                self.EAR_CLOSED_THRESH = self.baseline_ear * 0.75 
                self.is_calibrated = True
                print(f"✅ Đã học xong mắt! EAR chuẩn: {self.baseline_ear:.3f} -> Ngưỡng nhắm: {self.EAR_CLOSED_THRESH:.3f}")
            return self.timer_sleep # Đang học thì chưa báo động ngủ gật

        # BƯỚC 2: Sau khi học xong thì chạy logic phân tích như bình thường
        left_r = calculate_r_ratio(LEFT_BROW, LEFT_TOP, LEFT_BOTTOM, landmarks)
        right_r = calculate_r_ratio(RIGHT_BROW, RIGHT_TOP, RIGHT_BOTTOM, landmarks)
        r_current = (left_r + right_r) / 2.0
        self.r_buffer.append(r_current)
        
        is_blinking_now = False
        if len(self.r_buffer) == 3:
            delta_r = max(self.r_buffer) - min(self.r_buffer)
            if delta_r > self.t_blink:
                is_blinking_now = True 
                self.t_blink = (2.0 / 3.0) * delta_r 
        
        is_eyes_closed = False
        if (not is_blinking_now) and (ear < self.EAR_CLOSED_THRESH): 
            is_eyes_closed = True

        if is_eyes_closed:
            self.timer_sleep += dt
        else:
            self.timer_sleep = max(0.0, self.timer_sleep - dt * 2.0)
            
        return self.timer_sleep

    def draw_alert(self, frame, music_playing, music_path, pygame):
        # Hiển thị thông báo đang thu thập dữ liệu mắt để test cho ngầu
        if not self.is_calibrated:
            cv2.putText(frame, "DANG HOC MAT... VUI LONG NHIN THANG", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return music_playing

        if self.timer_sleep >= 8.0:
            cv2.putText(frame, "BAO DONG CAP DO MAX!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
            if not music_playing:
                try:
                    pygame.mixer.music.load(music_path)
                    pygame.mixer.music.play(-1)
                    music_playing = True
                except:
                    pass
        elif self.timer_sleep >= 1.5:
            cv2.putText(frame, "NGU GAT!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            if music_playing:
                pygame.mixer.music.stop()
                music_playing = False
        else:
            if music_playing:
                pygame.mixer.music.stop()
                music_playing = False
                
        return music_playing