import numpy as np
import cv2
from PIL import Image

class ImageDithering:
    def __open_image(name, img_scale = 7):
        img = cv2.imread(name)

        if img is None:
            print("Error: Could not open or find the image.")
            return

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        float_img = np.float32(gray_img)
        resized_img = cv2.resize(float_img, (float_img.shape[1] // img_scale, float_img.shape[0] // img_scale))

        return resized_img

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

    def __save_image(name, img):
        if img is None:
            print("Error: No image to save.")
            return

        if not name.lower().endswith('.png'):
            name = name.rsplit('.', 1)[0] + '.png' if '.' in name else name + '.png'

        pil_img = Image.fromarray(img)

        img_1bit = pil_img.convert("1")
        img_1bit.save(name, "PNG")
        
        print(f"Image saved as '{name}'")

    def processing(img_name, reducing_scale):
        img = ImageDithering.__open_image(img_name, reducing_scale)
        dithered = ImageDithering.__dither(img)
        #ImageDithering.__save_image("Dithered_Image", dithered)
        return dithered

ImageDithering.processing("carro.jpg", 10)