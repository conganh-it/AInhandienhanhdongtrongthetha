import cv2
import os

# Đường dẫn tới video
video_path = "E:/Program Files/nckh/dữ liệu/5468907656632.mp4"

# Thư mục lưu các ảnh tách ra
output_folder = 'images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Mở video bằng OpenCV
vidcap = cv2.VideoCapture(video_path)

# Lấy số khung hình mỗi giây của video
fps = int(vidcap.get(cv2.CAP_PROP_FPS))

# Biến đếm khung hình
count = 0
while True:
    success, image = vidcap.read()
    if not success:
        break

    # Kiểm tra xem có phải là khung hình ở mỗi giây hay không
    if count % fps == 0:
        # Lưu ảnh
        frame_filename = os.path.join(output_folder, f'frame{count // fps}.jpg')
        cv2.imwrite(frame_filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f'Lưu {frame_filename}')

    count += 1

vidcap.release()
print("Hoàn thành việc tách ảnh từ video.")
