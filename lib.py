import numpy as np
import cv2
from tqdm import tqdm
import subprocess

class Dithering:

    @staticmethod
    def apply_to_video(video_path, resolution_scale=7, fps=30):
        print("Starting script, this action may take some time...")
        frames = Dithering.__image_sequence(video_path)
        dithered_frames = []
        for frame in tqdm(frames, desc='Dithering frames', colour="green"):
            processed_frame = Dithering.__image_processing(frame, resolution_scale)
            dithered_frame = Dithering.__dither(processed_frame)
            dithered_frames.append(dithered_frame)
        Dithering.__save_frames(dithered_frames, "output", fps)
        #Dithering.__apply_sound(video_path)
        print("Script ran succesfully")

    @staticmethod
    def __save_frames(frames, name, fps):
        height, width = frames[0].shape[:2]
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{name}.mp4', codec, fps, (width, height))
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)

        out.release()

    @staticmethod
    def __image_sequence(video_path):

        frames = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        
        i = 0
        while True:
            i += 1
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        print("Frames succesfully accessed")
        cap.release()
        cv2.destroyAllWindows()

        return frames
    
    @staticmethod
    def __dither(img):

        img = img.copy().astype(np.float32)
        lines, columns = img.shape

        for l in range(lines):
            for c in range(columns):
                old_pixel = img[l, c]
                new_pixel = 255 if old_pixel > 127 else 0
                img[l, c] = new_pixel
                error = old_pixel - new_pixel

                if c + 1 < columns:
                    img[l, c + 1] += error * 7 / 16
                if l + 1 < lines:
                    if c > 0:
                        img[l + 1, c - 1] += error * 3 / 16
                    img[l + 1, c] += error * 5 / 16
                    if c + 1 < columns:
                        img[l + 1, c + 1] += error * 1 / 16

        return img.astype(np.uint8)
    
    @staticmethod
    def __image_processing(img, resolution_scale):
        to_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        to_float = np.float32(to_grey)
        resized_img = cv2.resize(to_float, (to_float.shape[1] // resolution_scale, to_float.shape[0] // resolution_scale))
        return resized_img
    
    @staticmethod
    def __apply_sound(source_video_path, output_path='output.mp4'):
        # Single ffmpeg command to copy video and re-encode audio
        command = [
            'ffmpeg',
            '-y',                   # Overwrite output
            '-i', source_video_path, # Input file
            '-c:v', 'copy',         # Copy video stream
            '-c:a', 'aac',          # Re-encode audio to AAC
            '-map', '0:v:0',        # Select video stream
            '-map', '0:a:0',        # Select audio stream
            '-shortest',            # Trim to shortest stream
            output_path
        ]
        subprocess.run(command, check=True)

    @staticmethod
    def open_image(name):
        img = cv2.imread(name)
        img = img.copy().astype(np.float32)
        
        if img is None:
            print("Error: Could not open or find the image.")
            return

        return img

    @staticmethod
    def terminal_dither(img_path, res_scale=14, white="-", black="*"):

        img = Dithering.open_image(img_path)
        img = Dithering.__image_processing(img, res_scale)
        img = Dithering.__dither(img)
        lines, columns = img.shape

        for l in range(lines):
            row = []
            for c in range(columns):
                row.append(black if img[l, c] == 255 else white)
            print(" ".join(map(str, row)))