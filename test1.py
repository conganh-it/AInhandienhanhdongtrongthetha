import os
import cv2
import mediapipe as mp
import pickle


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


def extract_pose_from_images(image_folder, output_file):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
    all_landmarks = {}

    for count, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        landmarks = extract_pose_from_image(image_path)
        if landmarks:
            all_landmarks[image_file] = landmarks
            print(f"Đã xử lý {image_file} ({count + 1}/{len(image_files)})")

    with open(output_file, 'wb') as f:
        pickle.dump(all_landmarks, f)
    print(f"Đã lưu tọa độ khung xương vào {output_file}")


if __name__ == "__main__":
    image_folder = "E:/Program Files/nckh/images" # Thay đường dẫn tới thư mục chứa ảnh của bạn
    output_file = "output_pose_data.pkl"
    extract_pose_from_images(image_folder, output_file)
