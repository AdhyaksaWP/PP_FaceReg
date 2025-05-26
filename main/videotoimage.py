import cv2
import os

def convert_video_to_30fps_and_extract_frames(video_path, output_dir, target_fps=30):
    # Buat folder output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Buka video asli
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps

    # Hitung total frame pada 30 FPS
    total_target_frames = int(duration * target_fps)

    # Ambil frame dan simpan sebagai gambar
    frame_idx = 0
    saved_frame = 0
    while cap.isOpened() and saved_frame < total_target_frames:
        # Posisi frame yang ingin diambil berdasarkan waktu
        time_sec = saved_frame / target_fps
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

        ret, frame = cap.read()
        if not ret:
            break

        # Simpan frame sebagai gambar
        frame_filename = os.path.join(output_dir, f"frame_{saved_frame:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        saved_frame += 1

    cap.release()
    print(f"Selesai! Total {saved_frame} frame disimpan di folder: {output_dir}")

# Contoh penggunaan
video_file = 'willi.mp4'  # Ganti dengan path video asli
output_folder = 'willi_per_frame'
convert_video_to_30fps_and_extract_frames(video_file, output_folder)
