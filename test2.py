import os
import cv2
import mediapipe as mp
import pickle
import numpy as np


def extract_pose_from_image(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    pose.close()

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = [
            landmarks[mp_pose.PoseLandmark.NOSE.value],  # Đỉnh đầu
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],  # Vai trái
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],  # Vai phải
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],  # Khuỷu tay trái
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],  # Khuỷu tay phải
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],  # Cổ tay trái
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],  # Cổ tay phải
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],  # Hông trái
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],  # Hông phải
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],  # Đầu gối trái
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],  # Đầu gối phải
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],  # Cổ chân trái
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]  # Cổ chân phải
        ]
        return [(lm.x, lm.y, lm.z) for lm in keypoints]
    return None


def compare_pose_data(pose_data_file, image_path):
    with open(pose_data_file, 'rb') as f:
        pose_data = pickle.load(f)

    image_landmarks = extract_pose_from_image(image_path)

    if image_landmarks is None:
        print("Không phát hiện khung xương trong tấm ảnh")
        return

    match_count = 0
    total_points = len(image_landmarks)

    for key in pose_data:
        sample_landmarks = pose_data[key]
        sample_landmarks = np.array(sample_landmarks)
        image_landmarks = np.array(image_landmarks)

        # Tính khoảng cách Euclidean giữa các điểm tương ứng
        distances = np.linalg.norm(sample_landmarks - image_landmarks, axis=1)

        # Ngưỡng sai số
        threshold = 0.5

        # Đếm số điểm trùng khớp
        match_count = np.sum(distances <= threshold)

    # Tính tỷ lệ trùng khớp
    match_percentage = (match_count / total_points) * 100

    # In kết quả
    if match_count == 0:
        print("Không trùng khớp")
    else:
        print(f"Tỉ lệ trùng khớp: {match_percentage:.2f}%")


if __name__ == "__main__":
    pose_data_file = "output_pose_data.pkl"  # Đường dẫn tới tệp dữ liệu tọa độ
    image_path = "E:/Program Files/nckh/images/5468907656632_000009.jpg"  # Đường dẫn tới tấm ảnh để so sánh
    compare_pose_data(pose_data_file, image_path)
