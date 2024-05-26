import socket

# Tạo socket TCP/IP
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Kết nối đến server ESP32
server_address = ('192.168.1.100', 80)  # Địa chỉ IP của ESP32 và cổng
sock.connect(server_address)

try:
    while True:
        message = "Hello from Python\n"
        sock.sendall(message.encode())  # Gửi tin nhắn đến ESP32

        data = sock.recv(1024).decode('utf-8').strip()  # Đọc phản hồi từ ESP32
        print("ESP32:", data)

finally:
    sock.close()
