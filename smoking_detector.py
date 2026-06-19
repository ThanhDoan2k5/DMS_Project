import cv2
from utils import MOUTH_INDICES, get_bounding_box, calculate_iomr

class SmokingDetector:
    def __init__(self):
        # Trả về ngưỡng 30% giao thoa vì test bằng webcam ngồi rất gần
        self.IOMR_SMOKING_THRESH = 0.30
        self.timer_smoking = 0.0

    def process(self, frame, landmarks, obj_boxes, dt):
        is_smoking_now = False
        mouth_box = get_bounding_box(MOUTH_INDICES, landmarks, frame.shape[1], frame.shape[0])
        
        for box in obj_boxes['smoke']:
            x1, y1, x2, y2, _ = box
            smoke_box = (x1, y1, x2, y2)
            if calculate_iomr(smoke_box, mouth_box) > self.IOMR_SMOKING_THRESH:
                is_smoking_now = True
                break

        if is_smoking_now: 
            self.timer_smoking += dt
        else: 
            # Phạt trừ thời gian nhanh gấp 5 lần để chống nhiễu hạt
            self.timer_smoking = max(0.0, self.timer_smoking - dt * 5.0)
            
        return self.timer_smoking

    def draw_alert(self, frame):
        # Trả về chuẩn phải ngậm liên tục 1.0 giây mới báo động
        if self.timer_smoking > 1.0:
            cv2.putText(frame, "CANH BAO: HUT THUOC", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)