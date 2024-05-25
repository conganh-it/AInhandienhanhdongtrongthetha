import cv2
import mediapipe as mp

# Khởi tạo MediaPipe pose và các công cụ vẽ
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Mở video
video_path = "E:/Program Files/nckh/dữ liệu/5468907517950.mp4"
cap = cv2.VideoCapture(video_path)

# Kiểm tra FPS của video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 0.1)  # Số lượng khung hình mỗi 0.5 giây

# Khởi tạo Pose
with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                  min_detection_confidence=0.5) as pose:
    frame_count = 0
    previous_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ xử lý mỗi frame_interval khung hình
        if frame_count % frame_interval == 0:
            # Chuyển đổi ảnh từ BGR sang RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Thực hiện pose estimation
            results = pose.process(image)

            if results.pose_landmarks:
                # Trích xuất tọa độ khung xương
                current_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

                # Kiểm tra xem các tọa độ hiện tại có giống với tọa độ trước đó không
                if current_landmarks != previous_landmarks:
                    previous_landmarks = current_landmarks

                    # Vẽ kết quả lên khung hình
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # In ra tọa độ của các điểm khung xương
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        print(f'ID: {id}, X: {cx}, Y: {cy}')

                    # Hiển thị khung hình
                    cv2.imshow('Pose Estimation', frame)

        frame_count += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
