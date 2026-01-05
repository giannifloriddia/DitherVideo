import numpy as np
import cv2
from tqdm import tqdm
import subprocess
import tempfile
import os

class Dithering:

    @staticmethod
    def apply_to_video(video_path, resolution_scale=7):
        """Process video and return bytes of the dithered video."""
        print("Starting script, this action may take some time...")
        frames, original_fps = Dithering.__image_sequence(video_path)

        fps = original_fps
        
        dithered_frames = []
        for frame in tqdm(frames, desc='Dithering frames', colour="green"):
            processed_frame = Dithering.__image_processing(frame, resolution_scale)
            dithered_frame = Dithering.__dither(processed_frame)
            dithered_frames.append(dithered_frame)
        
        video_bytes = Dithering.__save_frames_to_bytes(dithered_frames, fps)
        print("Script ran successfully")
        return video_bytes

    @staticmethod
    def __save_frames_to_bytes(frames, fps):
        """Save frames to a temporary file and return bytes."""
        height, width = frames[0].shape[:2]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, codec, fps, (width, height))
            
            for frame in frames:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame)
            
            out.release()
            
            # Read the file bytes
            with open(tmp_path, 'rb') as f:
                video_bytes = f.read()
            
            return video_bytes
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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
        """Extract frames from video and return frames with fps."""
        frames = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fps
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        print("Frames successfully accessed")
        cap.release()

        return frames, fps
    
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
        #Help here!
        pass

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