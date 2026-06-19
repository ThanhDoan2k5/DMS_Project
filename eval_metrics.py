import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

csv_path = 'dms_real_test.csv'

if not os.path.exists(csv_path):
    print(f"❌ Không tìm thấy file {csv_path}. Hãy chạy main.py để thu thập dữ liệu trước!")
    exit()

# Đọc dữ liệu thực tế từ file CSV
df = pd.read_csv(csv_path)

if len(df) == 0:
    print("❌ File CSV trống. Hãy ghi dữ liệu trước khi đánh giá!")
    exit()

y_true = df['ground_truth'].values
y_pred = df['prediction'].values
target_names = ['Normal', 'Drowsiness', 'Distraction', 'Smoking']

print("\n" + "="*60)
print(" 📊 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG THỰC TẾ (REAL DATA)")
print("="*60)
print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))
print("="*60)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.set(font_scale=1.2)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names,
            linewidths=1, linecolor='black')

plt.title('Ma trận nhầm lẫn (Confusion Matrix) - Thực nghiệm', fontsize=16, pad=20, weight='bold')
plt.ylabel('Nhãn thực tế (Ground Truth)', fontsize=14, weight='bold')
plt.xlabel('AI Dự đoán (Prediction)', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('Confusion_Matrix_Real.png', dpi=300)
print("✅ Đã xuất biểu đồ thực tế: 'Confusion_Matrix_Real.png'")
plt.show()