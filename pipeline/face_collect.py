import cv2
import mediapipe as mp
import os
import time

class FaceCollector:
    def __init__(self, output_dir='./dataset', duration=10, interval=1):
        self.output_dir = output_dir
        self.duration = duration
        self.interval = interval
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        os.makedirs(self.output_dir, exist_ok=True)

    def collect_faces(self):
        start_time = time.time()
        next_save_time = start_time
        image_count = 0

        while self.cap.isOpened() and (time.time() - start_time) < self.duration:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            current_time = time.time()
            if results.detections and current_time >= next_save_time:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    w_box = int(bbox.width * w)
                    h_box = int(bbox.height * h)

                    face_crop = frame[max(0, y):y + h_box, max(0, x):x + w_box]
                    if face_crop.size > 0:
                        filename = os.path.join(self.output_dir, f"face_{image_count}.jpg")
                        cv2.imwrite(filename, face_crop)
                        image_count += 1

                next_save_time += self.interval

            if results.detections:
                for detection in results.detections:
                    self.drawing.draw_detection(frame, detection)

            cv2.imshow('Face Collector', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()

if __name__ == "__main__":
    worker = FaceCollector()
    worker.collect_faces()