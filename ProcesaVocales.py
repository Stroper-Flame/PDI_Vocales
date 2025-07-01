import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

# --- FUNCIONES ---

def mascaras(imagen_gray):
    # Paso 1: Binarizar con Otsu + inversión
    _, binaria = cv2.threshold(imagen_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Paso 2: Apertura para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erosion = cv2.erode(binaria,kernel,iterations=5)
    limpia = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations=5)

    # Paso 3: Skeletonize (requiere imagen booleana)
    bin_bool = limpia > 0
    esqueleto = skeletonize(bin_bool).astype(np.uint8) * 255

    return binaria, limpia, esqueleto

def centrar_imagen(img, size=(100, 100)):
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros(size, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    recorte = img[y:y+h, x:x+w]
    escala = min(size[0] / w, size[1] / h)
    nueva_w, nueva_h = int(w * escala), int(h * escala)
    redimensionada = cv2.resize(recorte, (nueva_w, nueva_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros(size, dtype=np.uint8)
    offset_x = (size[0] - nueva_w) // 2
    offset_y = (size[1] - nueva_h) // 2
    canvas[offset_y:offset_y+nueva_h, offset_x:offset_x+nueva_w] = redimensionada
    return canvas

# --- PROCESAR Y GUARDAR ---
def procesarGuardar(ruta_entrada, ruta_salida):
    if not os.path.exists(ruta_salida):
        os.makedirs(ruta_salida)

    archivos = [f for f in os.listdir(ruta_entrada) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"[INFO] Procesando {len(archivos)} imágenes...")

    for nombre in archivos:
        path = os.path.join(ruta_entrada, nombre)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"No se pudo leer: {nombre}")
            continue

        binaria, limpia, esqueleto = mascaras(img)
        centrada = centrar_imagen(esqueleto)

        nombre_base = os.path.splitext(nombre)[0]
        cv2.imwrite(os.path.join(ruta_salida, nombre_base + "_binaria.png"), binaria)
        cv2.imwrite(os.path.join(ruta_salida, nombre_base + "_limpia.png"), limpia)
        cv2.imwrite(os.path.join(ruta_salida, nombre_base + "_esqueleto.png"), centrada)

        print(f"Guardado: {nombre_base}_*.png")

if __name__ == "__main__":
    ruta_entrada = r"C:\Users\delL-\Documents\UPIIT-IPN\Procesamiento Digital de Imagenes\Vocales"       
    ruta_salida = "VocalesPorcesadas"          
    procesarGuardar(ruta_entrada, ruta_salida)
