import cv2
import mediapipe as mp
import os
import csv
from unidecode import unidecode

# Khởi tạo MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_landmarks_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        return None

    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        # Đổi màu ảnh từ BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        if results.pose_landmarks:
            return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        return None

def save_landmarks_to_csv(landmarks_list, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for filename, landmarks in landmarks_list:
            flattened_landmarks = [coord for lm in landmarks for coord in lm]
            writer.writerow([filename] + flattened_landmarks)

def load_landmarks_from_csv(input_file):
    landmarks_list = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            filename = row[0]
            landmarks = [(float(row[i]), float(row[i + 1]), float(row[i + 2])) for i in range(1, len(row), 3)]
            landmarks_list.append((filename, landmarks))
    return landmarks_list

# Đường dẫn đến thư mục chứa các ảnh
images_folder = r'E:/Program Files/nckh/images'

# Trích xuất đặc trưng từ các ảnh
landmarks_list = []
for filename in os.listdir(images_folder):
    # Chuyển đổi tên file sang ASCII để tránh lỗi
    safe_filename = unidecode(filename)
    image_path = os.path.join(images_folder, safe_filename)
    landmarks = extract_landmarks_from_image(image_path)
    if landmarks:
        landmarks_list.append((filename, landmarks))

# Lưu danh sách đặc trưng để so sánh sau này
output_file = 'landmarks_list.csv'
save_landmarks_to_csv(landmarks_list, output_file)

# Tải danh sách đặc trưng đã lưu để kiểm tra
loaded_landmarks_list = load_landmarks_from_csv(output_file)
for filename, landmarks in loaded_landmarks_list:
    print(f"File: {filename}, Landmarks: {landmarks[:5]}...")  # Hiển thị 5 landmarks đầu tiên để kiểm tra
