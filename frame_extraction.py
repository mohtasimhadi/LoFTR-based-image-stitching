import cv2
import os
import glob

def extract_frames_from_hvec(video_path, output_folder):
    print(video_path)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    os.makedirs(output_folder, exist_ok=True)
    frames = []
    i = 1

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    video.release()

    for frame in frames:
        cv2.imwrite(f"{output_folder}/{i:04}.png", frame)
        print(output_folder, i)
        i += 1

def process_input_folders(input_dir):
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            hvec_files = glob.glob(os.path.join(folder_path, "*.h265"))
            for hvec_file in hvec_files:
                video_name = os.path.splitext(os.path.basename(hvec_file))[0]
                output_folder = os.path.join(folder_path, f"{video_name}_frames")
                extract_frames_from_hvec(hvec_file, output_folder)

input_directory = "input/02/2023-11-07_19-59-56_184430105185341300_color.h265"
output_driectory = "out/02/18443010518B880E00"
extract_frames_from_hvec(input_directory, output_driectory)
