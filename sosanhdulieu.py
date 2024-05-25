import cv2
import mediapipe as mp
import pickle
import os

def compare_pose_with_video(video_path, pose_data_file):
    if not os.path.exists(pose_data_file):
        print(f"Error: Pose data file {pose_data_file} does not exist.")
        return

    with open(pose_data_file, 'rb') as f:
        pose_data = pickle.load(f)

    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 2)

    match_count = 0
    total_points = 0

    with mp_pose as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    total_points += 1
                    pose_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                    for image_filename, image_landmarks in pose_data.items():
                        if pose_landmarks == image_landmarks:
                            match_count += 1

            frame_count += 1

    cap.release()
    match_percentage = (match_count / total_points) * 100
    print(f"Matching percentage: {match_percentage:.2f}%")

if __name__ == "__main__":
    video_path = "E:/Program Files/nckh/dữ liệu/5468907656632.mp4"
    pose_data_file = "pose_data.pkl"
    compare_pose_with_video(video_path, pose_data_file)
