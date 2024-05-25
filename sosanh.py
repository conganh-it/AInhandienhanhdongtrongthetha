import cv2
import mediapipe as mp
import pickle
import os


def extract_pose_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return None

    mp_pose = mp.solutions.pose
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image from {image_path}")
        return None

    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            image_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Pose Estimation on Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return image_landmarks
    return None


def compare_pose_data(video_data_file, image_landmarks):
    if not os.path.exists(video_data_file):
        print(f"Error: Video data file {video_data_file} does not exist.")
        return

    with open(video_data_file, 'rb') as f:
        video_pose_data = pickle.load(f)

    match_count = 0
    total_points = 0

    for frame_count, video_landmarks in video_pose_data:
        total_points += len(video_landmarks)
        for video_point, image_point in zip(video_landmarks, image_landmarks):
            if video_point == image_point:
                match_count += 1

    match_percentage = (match_count / total_points) * 100
    print(f"Tỉ lệ trùng khớp: {match_percentage:.2f}%")


if __name__ == "__main__":
    image_path = "E:/Program Files/nckh/images/5468907507962_000010.jpg"
    video_data_file = 'video_pose_data.pkl'

    image_landmarks = extract_pose_from_image(image_path)
    if image_landmarks:
        compare_pose_data(video_data_file, image_landmarks)
    else:
        print("Không thể trích xuất tọa độ khung xương từ ảnh.")
