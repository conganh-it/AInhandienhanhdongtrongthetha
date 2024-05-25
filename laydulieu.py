import cv2
import mediapipe as mp
import pickle
import os

def extract_pose_from_images(image_folder, output_file):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    pose_data = {}

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue

            with mp_pose as pose:
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    pose_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                    pose_data[filename] = pose_landmarks

    with open(output_file, 'wb') as f:
        pickle.dump(pose_data, f)

    print(f"Pose data extracted from images in {image_folder} and saved to {output_file}")

if __name__ == "__main__":
    image_folder = "E:/Program Files/nckh/images"
    output_file = "pose_data.pkl"
    extract_pose_from_images(image_folder, output_file)
