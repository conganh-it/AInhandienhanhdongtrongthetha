import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from PIL import Image, ImageTk
import time

class PoseComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Comparator")
        self.root.geometry("800x600")
        self.pose_data_file = None
        self.camera_index = 0
        self.cap = None
        self.start_time = None
        self.elapsed_time = 0
        self.running = False
        self.match_percentages = []

        self.canvas_camera = tk.Canvas(root, width=640, height=480, bg="blue")
        self.canvas_camera.grid(row=0, column=0, padx=20, pady=20, columnspan=4)

        self.time_label = tk.Label(root, text="Thời gian: 0s", bg="blue", fg="white", font=("Arial", 16))
        self.time_label.grid(row=1, column=0, padx=10, pady=10)

        self.match_label = tk.Label(root, text="Tỉ lệ trùng khớp:", bg="blue", fg="white", font=("Arial", 16))
        self.match_label.grid(row=1, column=1, padx=10, pady=10, columnspan=3)

        self.file_button = tk.Button(root, text="Chọn file", command=self.choose_file, bg="blue", fg="white",
                                     font=("Arial", 12))
        self.file_button.grid(row=2, column=0, padx=10, pady=10)

        self.camera_button = tk.Button(root, text="Chọn camera", command=self.choose_camera, bg="blue", fg="white",
                                       font=("Arial", 12))
        self.camera_button.grid(row=2, column=1, padx=10, pady=10)

        self.start_button = tk.Button(root, text="Bắt đầu", command=self.start_camera, bg="blue", fg="white",
                                      font=("Arial", 12))
        self.start_button.grid(row=2, column=2, padx=10, pady=10)

        self.stop_button = tk.Button(root, text="Kết thúc", command=self.stop_camera, bg="blue", fg="white",
                                     font=("Arial", 12))
        self.stop_button.grid(row=2, column=3, padx=10, pady=10)

        self.exit_button = tk.Button(root, text="Thoát", command=root.quit, bg="blue", fg="white", font=("Arial", 12))
        self.exit_button.grid(row=3, column=0, columnspan=4, pady=10)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                                      min_detection_confidence=0.5)

    def choose_file(self):
        self.pose_data_file = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if self.pose_data_file:
            messagebox.showinfo("Thông báo", "Đã chọn file dữ liệu")

    def choose_camera(self):
        self.camera_index = simpledialog.askinteger("Chọn camera", "Nhập số camera:")
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được camera")
        else:
            messagebox.showinfo("Thông báo", f"Đã chọn camera {self.camera_index}")

    def extract_pose_from_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [
                landmarks[self.mp_pose.PoseLandmark.NOSE.value],  # Đỉnh đầu
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],  # Vai trái
                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],  # Vai phải
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],  # Khuỷu tay trái
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],  # Khuỷu tay phải
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value],  # Cổ tay trái
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value],  # Cổ tay phải
                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],  # Hông trái
                landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],  # Hông phải
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],  # Đầu gối trái
                landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],  # Đầu gối phải
                landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],  # Cổ chân trái
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]  # Cổ chân phải
            ]
            return [(lm.x, lm.y, lm.z) for lm in keypoints]
        return None

    def compare_pose_data(self, image_landmarks):
        if not self.pose_data_file:
            return 0.0

        with open(self.pose_data_file, 'rb') as f:
            pose_data = pickle.load(f)

        if image_landmarks is None:
            return 0.0

        match_count = 0
        total_points = len(image_landmarks)

        for key in pose_data:
            sample_landmarks = pose_data[key]
            sample_landmarks = np.array(sample_landmarks)
            image_landmarks = np.array(image_landmarks)

            # Tính khoảng cách Euclidean giữa các điểm tương ứng
            distances = np.linalg.norm(sample_landmarks - image_landmarks, axis=1)

            # Ngưỡng sai số
            threshold = 0.1

            # Đếm số điểm trùng khớp
            match_count += np.sum(distances <= threshold)

        match_percentage = (match_count / total_points) * 100
        return match_percentage

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Chưa chọn camera hoặc không thể mở được camera")
            return
        if not self.running:
            self.running = True
            self.start_time = time.time() - self.elapsed_time
            self.update_time()
            self.update_frame()

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.canvas_camera.delete("all")
        self.match_label.config(text="Tỉ lệ trùng khớp:")

        # Calculate the average match percentage
        if self.match_percentages:
            avg_match_percentage = np.mean(self.match_percentages)
        else:
            avg_match_percentage = 0.0

        # Display the average match result in a message box
        if avg_match_percentage > 50:
            messagebox.showinfo("Kết quả", f"Tỉ lệ trùng khớp: {avg_match_percentage:.2f}%\nĐộng tác: Đạt")
        else:
            messagebox.showinfo("Kết quả", f"Tỉ lệ trùng khớp: {avg_match_percentage:.2f}%\nĐộng tác: Không đạt")

    def update_time(self):
        if self.running:
            self.elapsed_time = time.time() - self.start_time
            self.time_label.config(text=f"Thời gian: {int(self.elapsed_time)}s")
            self.root.after(1000, self.update_time)

    def update_frame(self):
        if self.cap and self.running:
            ret, frame = self.cap.read()
            if ret:
                image_landmarks = self.extract_pose_from_image(frame)
                match_percentage = self.compare_pose_data(image_landmarks)
                self.match_percentages.append(match_percentage)
                self.match_label.config(text=f"Tỉ lệ trùng khớp: {match_percentage:.2f}%")

                if image_landmarks:
                    frame = self.draw_skeleton(frame, image_landmarks)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas_camera.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas_camera.image = imgtk

        if self.running:
            self.root.after(10, self.update_frame)

    def draw_skeleton(self, frame, landmarks):
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
            (1, 7), (2, 8), (7, 9), (8, 10), (9, 11), (10, 12)
        ]

        h, w, _ = frame.shape
        for (x, y, z) in landmarks:
            cv2.circle(frame, (int(x * w), int(y * h)), 5, (0, 0, 255), -1)

        for start, end in connections:
            x1, y1, z1 = landmarks[start]
            x2, y2, z2 = landmarks[end]
            cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)

        return frame


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseComparatorApp(root)
    root.mainloop()
