import cv2
import mediapipe as mp
import pickle


def extract_pose_from_video(video_path, output_file, interval=0.5):
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    pose_data = []
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                      min_detection_confidence=0.5) as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                if results.pose_landmarks:
                    current_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                    pose_data.append((frame_count, current_landmarks))

            frame_count += 1

    cap.release()
    with open(output_file, 'wb') as f:
        pickle.dump(pose_data, f)
    print(f"Đã lưu tọa độ khung xương từ video vào {output_file}")


if __name__ == "__main__":
    video_path = "E:/Program Files/nckh/dữ liệu/5468907517950.mp4"
    output_file = 'video_pose_data.pkl'

    extract_pose_from_video(video_path, output_file)
