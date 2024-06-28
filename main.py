from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
import socket
import threading
import time

app = Flask(__name__)
model = YOLO('best (3).pt')

# Tỷ lệ chuyển đổi từ pixel sang centimet
PIXEL_TO_CM = 0.0250

# Cấu hình máy chủ socket
PORT = 5000
SERVER = "192.168.111.125"  # Thay thế bằng địa chỉ IP của server của bạn
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
distance_realtime = 0
last_sent_time = 0  # Thời điểm gửi lệnh gần nhất

# Ứng dụng Flask
def send_message_to_server(message):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(ADDR)
            client_socket.send(message.encode(FORMAT))
            print(f"Gửi tin nhắn tới server: {message}")
            response = client_socket.recv(1024).decode(FORMAT)
            print(f"Nhận phản hồi từ server: {response}")
            return response
    except Exception as e:
        print(f"Gửi tin nhắn tới server thất bại: {e}")
        return None

def generate_frames():
    global distance_realtime, last_sent_time
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Lật ảnh theo chiều ngang
        if not success:
            break
        else:
            # Xác định vùng quan tâm (ROI)
            x1, y1, x2, y2 = 200, 0, 600, frame.shape[0]
            frame_cut = frame[y1:y2, x1:x2]

            # Vẽ hình chữ nhật trên khung hình gốc
            start_point = (x1, y1)
            end_point = (x2, y2)
            color = (255, 255, 0)
            thickness = 2
            frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

            # Chạy YOLOv8 tracking trên ROI
            results = model.track(frame_cut, persist=True)
            person_centers = []
            garbage_centers = []

            for result in results:
                boxes = result.boxes.numpy()
                for box in boxes:
                    x11 = int(box.xyxy[0][0]) + x1
                    y11 = int(box.xyxy[0][1]) + y1
                    x21 = int(box.xyxy[0][2]) + x1
                    y21 = int(box.xyxy[0][3]) + x1
                    frame = cv2.rectangle(frame, (x11, y11), (x21, y21), color, thickness)

                    # Lấy chỉ số nhãn lớp
                    class_idx = int(box.cls[0])

                    # Lấy tên lớp từ model
                    class_name = model.names[class_idx]

                    # Hiển thị tên lớp
                    location = (x11, y11 - 5)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    frame = cv2.putText(frame, class_name, location, font, fontScale, color, thickness, cv2.LINE_AA)

                    # Tính toán điểm trung tâm
                    center_x = (x11 + x21) // 2
                    center_y = (y11 + y21) // 2

                    # Vẽ điểm trung tâm và lưu tọa độ cho person và garbage
                    if class_name == "person":
                        person_centers.append((center_x, center_y))
                        frame = cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    elif class_name == "garbage":
                        garbage_centers.append((center_x, center_y))
                        frame = cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Tính khoảng cách giữa tất cả các điểm trung tâm của người và điểm trung tâm của rác
            for person_center in person_centers:
                for garbage_center in garbage_centers:
                    distance_px = np.linalg.norm(np.array(person_center) - np.array(garbage_center))
                    distance_cm = distance_px * PIXEL_TO_CM  # Chuyển đổi sang centimet
                    midpoint = (
                        (person_center[0] + garbage_center[0]) // 2, (person_center[1] + garbage_center[1]) // 2)
                    frame = cv2.putText(frame, f'{distance_cm:.2f} cm', midpoint, font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

                    # In khoảng cách
                    print(f"Khoảng cách giữa người và rác: {distance_cm:.2f} cm")

                    # Cập nhật khoảng cách thời gian thực bằng mét
                    distance_realtime = distance_cm / 100

                    # Kiểm tra nếu khoảng cách nhỏ hơn 10cm, gửi lệnh mở thùng rác mỗi 10 giây
                    current_time = time.time()
                    if distance_realtime < 0.1 and current_time - last_sent_time >= 10:
                        send_message_to_server("Mở thùng rác\n")
                        last_sent_time = current_time

                    # Vẽ đường giữa các điểm trung tâm
                    frame = cv2.line(frame, person_center, garbage_center, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
