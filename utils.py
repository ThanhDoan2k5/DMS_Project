import math
import numpy as np

# Các chỉ số mắt và miệng từ MediaPipe Face Mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_BROW = 105; LEFT_TOP = 159; LEFT_BOTTOM = 145

RIGHT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_BROW = 334; RIGHT_TOP = 386; RIGHT_BOTTOM = 374

MOUTH_INDICES = [61, 291, 0, 17]

def euclidean_distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y]) 

def calculate_ear(eye_indices, landmarks): 
    p = [landmarks.landmark[i] for i in eye_indices]
    vertical_1 = euclidean_distance(p[1], p[5])
    vertical_2 = euclidean_distance(p[2], p[4])
    horizontal = euclidean_distance(p[0], p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_r_ratio(brow_idx, top_idx, bottom_idx, landmarks): 
    p_brow = landmarks.landmark[brow_idx]
    p_top = landmarks.landmark[top_idx]
    p_bottom = landmarks.landmark[bottom_idx]
    d0 = euclidean_distance(p_brow, p_top)    
    d1 = euclidean_distance(p_top, p_bottom)  
    if d1 < 0.001: d1 = 0.001 
    return d0 / d1

def calculate_iomr(boxA, boxB):
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])
    
    if x_right < x_left or y_bottom < y_top: return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return intersection_area / boxB_area if boxB_area > 0 else 0

def get_bounding_box(indices, landmarks, w, h):
    x_coords = [int(landmarks.landmark[i].x * w) for i in indices]
    y_coords = [int(landmarks.landmark[i].y * h) for i in indices]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))