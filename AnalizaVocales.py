import cv2
import numpy as np
import os

def centrarImagen(binaria, size=(100, 100)):
    coords = cv2.findNonZero(binaria)
    if coords is None:
        return np.zeros(size, dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    recorte = binaria[y:y+h, x:x+w]

    escala = min(size[0] / w, size[1] / h)
    nueva_w, nueva_h = int(w * escala), int(h * escala)
    letra_resized = cv2.resize(recorte, (nueva_w, nueva_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros(size, dtype=np.uint8)
    offset_x = (size[0] - nueva_w) // 2
    offset_y = (size[1] - nueva_h) // 2
    canvas[offset_y:offset_y+nueva_h, offset_x:offset_x+nueva_w] = letra_resized

    return canvas

def limpiarImagen(binaria):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erosion = cv2.erode(binaria,kernel,iterations=1)
    limpia = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations=1)
    return limpia

def comparar(imagen_proc, banco):
    imagen_proc = centrarImagen(imagen_proc)

    mejor_score = float('inf')
    mejor_letra = '?'

    for letra, ref in banco:
        ref_proc = centrarImagen(ref)
        diff = cv2.bitwise_xor(imagen_proc, ref_proc)
        score = np.sum(diff) // 255  # Cada diferencia vale 255, normalizamos

        if score < mejor_score:
            mejor_score = score
            mejor_letra = letra

    return mejor_letra, mejor_score

def cargarVocalesProcesadas(ruta_banco, tipo_archivo):
    banco = []
    for archivo in os.listdir(ruta_banco):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')) and tipo_archivo in archivo.lower():
            partes = archivo.split('_')
            if len(partes) < 2:
                continue

            vocal = partes[0].upper()
            tipo_letra = partes[1]

            if "Ma" in tipo_letra:
                letra = vocal.upper()
            elif "Mi" in tipo_letra:
                letra = vocal.lower()
            else:
                continue

            path = os.path.join(ruta_banco, archivo)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            banco.append((letra, img))

    print(f"Imagenes cargadas con {len(banco)} vocales del tipo ({tipo_archivo}).")
    return banco

def tomarLetra(banco):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("S para salir, C para capturar", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            limpia = limpiarImagen(binaria)

            cv2.imshow("Preprocesada", limpia)

            letra, score = comparar(limpia, banco)
            tipo = "Mayúscula" if letra.isupper() else "Minúscula"
            print(f"Se detectó: {letra} ({tipo})")

        elif key == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tipo = 'limpia'  
    ruta_banco = r"C:\Users\delL-\Documents\UPIIT-IPN\Procesamiento Digital de Imagenes\VocalesPorcesadas"
    banco = cargarVocalesProcesadas(ruta_banco, tipo_archivo=tipo)
    tomarLetra(banco)
