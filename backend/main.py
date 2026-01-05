from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from image_dithering import ImageDithering
from lib import Dithering
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/dither_img")
async def dither_image(image: UploadFile = File(...), scaling_size: int = Form(7)):
    try:
        # Read uploaded file
        contents = await image.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Could not decode image"}
        
        dithered_img = ImageDithering.processing(img, scaling_size)
        
        # Save to buffer
        buf = BytesIO()
        dithered_img.save(buf, format='PNG')
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.post("/dither_video")
async def dither_video(video: UploadFile = File(...), scaling_size: int = Form(7)):
    try:
        # Read uploaded file
        contents = await video.read()
        
        # Save to temporary file (OpenCV needs a file path)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Process video
            video_bytes = Dithering.apply_to_video(tmp_path, resolution_scale=scaling_size)
            
            # Return video as streaming response
            buf = BytesIO(video_bytes)
            buf.seek(0)
            
            return StreamingResponse(
                buf, 
                media_type="video/mp4",
                headers={"Content-Disposition": "attachment; filename=dithered_video.mp4"}
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
