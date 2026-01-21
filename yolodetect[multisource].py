#ffmpeg -hide_banner  -i "rtsp://fayani:12121976@192.168.0.164:554/stream2" -f image2pipe  -vf fps=1 pipe:1
#cap = cv2.VideoCapture(0)
import cv2 # Import OpenCV Library
from ultralytics import YOLO # Import Ultralytics Package
import threading # Threading module import
import time

# Define the video files for the trackers
video_file1 = 'rtsp://fayani:12121976@192.168.0.164:554/stream2' # Video file path
video_file2 = 0  # WebCam Path

#prev_time = 0 
# Inisialisasi variabel waktu untuk interval simpan
#last_save_time = time.time()

# Load the YOLOv8 models
model1 = YOLO('/home/pi/deepface/yolov11n-face_full_integer_quant_edgetpu.tflite', task='detect') # YOLO Model
model2 = YOLO('/home/pi/deepface/yolov11n-face_full_integer_quant_edgetpu.tflite', task='detect') # YOLO Model

def run_tracker_in_thread(filename, model, file_index):
    """
    This function is designed to run a video file or webcam stream
    concurrently with the YOLOv8 model, utilizing threading.

    - filename: The path to the video file or the webcam/external
    camera source.
    - model: The file path to the YOLOv8 model.
    - file_index: An argument to specify the count of the
    file being processed.
    """

    video = cv2.VideoCapture(filename)  # Read the video file
    new_width = 360  #4:3
    new_height = 270
    new_dimensions = (new_width, new_height)
    last_save_time = time.time()

    while True:
        ret, gbr = video.read()  # Read the video frames
        frame = cv2.resize(gbr, new_dimensions, interpolation=cv2.INTER_LINEAR)
        curr_time = time.time()
        #fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        #prev_time = curr_time

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        #results = model.track(frame, persist=True, conf=0.2, line_width=2, verbose=False, show_labels=True, imgsz=256, device='tpu')
        results = model.predict(frame, conf=0.2, line_width=2, show_labels=True, verbose=False, stream_buffer=True, imgsz=256, device='tpu')
        res_plotted = results[0].plot()
        #cv2.imshow("Tracking_Stream_"+str(file_index), res_plotted)
        # Logika Penyimpanan Otomatis Tiap 1 Detik
        if curr_time - last_save_time >= 1.0:
            # Menggunakan timestamp agar file tidak menimpa satu sama lain
            #file_path = f"/home/pi/deepface/output_{int(curr_time)}.jpg"
            file_path = f"/home/pi/deepface/output_{int(file_index)}.jpg"
            cv2.imwrite(file_path, res_plotted)
            print(f'Gambar disimpan: {file_path}')
            last_save_time = curr_time  # Reset waktu simpan terakhir

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()

# Create the tracker thread

# Thread used for the video file
tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file1, model1, 1),
                                   daemon=True)

# Thread used for the webcam
tracker_thread2 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file2, model2, 2),
                                   daemon=True)

# Start the tracker thread

# Start thread that run video file
tracker_thread1.start()

# Start thread that run webcam
tracker_thread2.start()

# Wait for the tracker thread to finish

# Tracker thread 1
tracker_thread1.join()

# Tracker thread 2
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
