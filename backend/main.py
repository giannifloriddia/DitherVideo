from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from image_dithering import ImageDithering
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

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
