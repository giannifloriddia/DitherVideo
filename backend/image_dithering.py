import numpy as np
import cv2
from PIL import Image


class ImageDithering:
    
    @staticmethod
    def __open_image(name, img_scale=7):
        if isinstance(name, str):
            img = cv2.imread(name)
        else:
            img = name
        
        if img is None:
            print("Error: Could not open or find the image.")
            return None
        
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        float_img = np.float32(gray_img)
        
        resized_img = cv2.resize(
            float_img, 
            (float_img.shape[1] // img_scale, float_img.shape[0] // img_scale)
        )
        
        return resized_img

    @staticmethod
    def __dither(img):
        img = img.copy().astype(np.float32)
        linhas, colunas = img.shape

        for l in range(linhas):
            for c in range(colunas):
                pixelAntigo = img[l, c]
                pixelNovo = 255 if pixelAntigo > 127 else 0
                img[l, c] = pixelNovo
                erro = pixelAntigo - pixelNovo

                if c + 1 < colunas:
                    img[l, c + 1] += erro * 7 / 16
                if l + 1 < linhas:
                    if c > 0:
                        img[l + 1, c - 1] += erro * 3 / 16
                    img[l + 1, c] += erro * 5 / 16
                    if c + 1 < colunas:
                        img[l + 1, c + 1] += erro * 1 / 16

        return img.astype(np.uint8)

    @staticmethod
    def processing(img_input, reducing_scale=7):
        img = ImageDithering.__open_image(img_input, reducing_scale)
        if img is None:
            raise ValueError("Could not process image")
        dithered_img = ImageDithering.__dither(img)
        # Convert to PIL and make 1-bit
        pil_img = Image.fromarray(dithered_img)
        img_1bit = pil_img.convert("1")
        return img_1bit