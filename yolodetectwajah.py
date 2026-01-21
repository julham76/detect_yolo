#ffmpeg -hide_banner  -i "rtsp://fayani:12121976@192.168.0.164:554/stream2" -f image2pipe  -vf fps=1 pipe:1
from ultralytics import YOLO
import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "model_s", "my_model.pt")
#MODEL_PATH = os.path.join(BASE_DIR, "model_s", "best_full_integer_quant_edgetpu.tflite")
#model = YOLO('my_coco128_full_integer_quant_edgetpu.tflite', task='segment')
model = YOLO('yolov11n-face_full_integer_quant_edgetpu.tflite', task='detect')

#cap = cv2.VideoCapture('https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/chunklist_w221624478.m3u8')
cap = cv2.VideoCapture(0)

prev_time = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    results = model.predict(frame, conf=0.2, line_width=2, show_labels=True, verbose=False, imgsz=256, stream_buffer=True, device='tpu')
    annotated_frame = results[0].plot()

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

    cv2.imshow("YOLOv11 Realtime Detection PPE", annotated_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
