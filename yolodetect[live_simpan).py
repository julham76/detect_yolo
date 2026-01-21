#ffmpeg -hide_banner  -i "rtsp://fayani:12121976@192.168.0.164:554/stream2" -f image2pipe  -vf fps=1 pipe:1
from ultralytics import YOLO
import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model_s", "my_model.pt")
#MODEL_PATH = os.path.join(BASE_DIR, "model_s", "best_full_integer_quant_edgetpu.tflite")
#model = YOLO('my_coco128_full_integer_quant_edgetpu.tflite', task='segment')
model = YOLO('/home/pi/deepface/yolov11n-face_full_integer_quant_edgetpu.tflite', task='detect')

#cap = cv2.VideoCapture('https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/chunklist_w221624478.m3u8')
cap = cv2.VideoCapture(0)

prev_time = 0 
# Inisialisasi variabel waktu untuk interval simpan
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Inference menggunakan model
    results = model.predict(frame, conf=0.2, line_width=2, show_labels=True, verbose=False, stream_buffer=True, imgsz=256, device='tpu')
    annotated_frame = results[0].plot()

    # Logika Penyimpanan Otomatis Tiap 1 Detik
    if curr_time - last_save_time >= 1.0:
        # Menggunakan timestamp agar file tidak menimpa satu sama lain
        #file_path = f"/home/pi/deepface/output_{int(curr_time)}.jpg"
        file_path = f"/home/pi/deepface/output.jpg"
        cv2.imwrite(file_path, annotated_frame)
        print(f'Gambar disimpan: {file_path}')
        last_save_time = curr_time  # Reset waktu simpan terakhir

    # Menampilkan FPS pada frame
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),                   
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),                 
        2,
        cv2.LINE_AA
    )

    # Menampilkan hasil (opsional)
    #cv2.imshow("YOLOv11 Realtime Detection", annotated_frame)

    # Tetap sediakan tombol 'q' untuk berhenti secara manual
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
