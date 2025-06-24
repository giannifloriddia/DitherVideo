import numpy as np
import cv2

def open_image(name, img_scale = 7):
    img = cv2.imread(name)
    
    if img is None:
        print("Error: Could not open or find the image.")
        return
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    float_img = np.float32(gray_img)
    
    resized_img = cv2.resize(float_img, (float_img.shape[1] // img_scale, float_img.shape[0] // img_scale))

    return resized_img

def show_image(name, img):
    if img is None:
        print("Error: No image to display.")
        return
    
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(name, img):
    if img is None:
        print("Error: No image to save.")
        return
    
    cv2.imwrite(name, img)
    print(f"Image saved as '{name}'")

def dither(img):

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

def show_and_dither(name, scale = 7):
    if name is None:
        print("Error: No image to display.")
        return
    
    img = open_image(name, scale)
    dithered_img = dither(img)

    new_name = "Dithered_" + name

    show_image(new_name, dithered_img)

    return new_name, dithered_img

#show_and_dither("carro.jpg")
#name, perfil = show_and_dither("titi.jpg")

#save_image(name, perfil)

#show_and_dither("underTheSea.jpg", 10)
#show_and_dither("couple.jpg", 14)

show_and_dither("prof.png", 3)