
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Orden alfabetico de las carpetas del dataset (asi las ordena Keras al entrenar)
CLASES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

RUTA_MODELO = "modelo_residuos.keras"
_MODELO = None  # cache: el modelo se carga una sola vez


def cargar_modelo(ruta=RUTA_MODELO):
    """Carga modelo_residuos.keras (una sola vez) y lo guarda en memoria."""
    global _MODELO
    if _MODELO is None:
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No encuentro '{ruta}' en {os.getcwd()}. "
                "Ejecuta la celda modelo.save('modelo_residuos.keras') o copia el archivo aqui."
            )
        import tensorflow as tf
        from tensorflow.keras.applications.efficientnet import preprocess_input

        objetos = {"preprocess_input": preprocess_input}
        try:
            # El modelo tiene una capa Lambda(preprocess_input): Keras 3 pide safe_mode=False
            _MODELO = tf.keras.models.load_model(
                ruta, compile=False, safe_mode=False, custom_objects=objetos
            )
        except TypeError:  # versiones antiguas de Keras no tienen safe_mode
            _MODELO = tf.keras.models.load_model(
                ruta, compile=False, custom_objects=objetos
            )
    return _MODELO


def predecir_imagen(ruta_imagen, modelo=None, clases=CLASES, mostrar=True):
    """
    Clasifica una imagen y devuelve (clase, confianza).
    Muestra la imagen con la clase y la confianza, igual que el notebook original.

    modelo: opcional. Si ya tienes `modelo` en el notebook, pasalo: predecir_imagen(ruta, modelo=modelo)
            Si no, se carga modelo_residuos.keras.
    """
    if modelo is None:
        modelo = cargar_modelo()

    # Se redimensiona en bilineal, igual que image_dataset_from_directory al entrenar.
    # (El modelo ya incluye su propio preprocess_input, por eso NO se divide entre 255.)
    img = Image.open(ruta_imagen).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)

    probs = modelo.predict(arr, verbose=0)[0]
    indice = int(np.argmax(probs))
    clase = clases[indice]
    confianza = np.float32(probs[indice])

    if mostrar:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Clase: {clase}\nConfianza: {confianza * 100:.2f}%", fontsize=14)
        plt.axis("off")
        plt.show()

    print(f"Clase detectada: {clase}")
    print(f"Confianza: {confianza * 100:.2f}%")
    return clase, confianza


def segmentar_objeto(img_bgr, rect=None, iteraciones=5):
    """
    Separa el objeto principal del fondo con GrabCut.
    rect = (x, y, ancho, alto) que encierra el objeto. Si es None se usa la zona central.
    Devuelve (mascara 0/1, bounding box (x, y, ancho, alto)).
    """
    alto, ancho = img_bgr.shape[:2]

    # Trabajamos en una copia pequena: es mucho mas rapido y el resultado es igual de util
    escala = 600 / max(alto, ancho)
    chica = cv2.resize(img_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    chica = cv2.GaussianBlur(chica, (5, 5), 0)  # suaviza ruido antes de segmentar
    h, w = chica.shape[:2]

    if rect is None:
        rect = (int(ancho * 0.10), int(alto * 0.05), int(ancho * 0.80), int(alto * 0.90))
    x, y, rw, rh = [int(v * escala) for v in rect]

    mascara = np.zeros((h, w), np.uint8)
    fondo, frente = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(chica, mascara, (x, y, rw, rh), fondo, frente, iteraciones, cv2.GC_INIT_WITH_RECT)

    binaria = np.where((mascara == cv2.GC_FGD) | (mascara == cv2.GC_PR_FGD), 1, 0).astype("uint8")

    # Limpieza: quitar manchas chicas y quedarse con el objeto mas grande
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise ValueError("No se encontro ningun objeto. Prueba pasando rect=(x, y, ancho, alto).")
    mayor = max(contornos, key=cv2.contourArea)
    binaria = np.zeros_like(binaria)
    cv2.drawContours(binaria, [mayor], -1, 1, thickness=-1)

    # Volver al tamano original de la foto
    binaria = cv2.resize(binaria, (ancho, alto), interpolation=cv2.INTER_NEAREST)
    bx, by, bw, bh = cv2.boundingRect(mayor)
    bbox = tuple(int(v / escala) for v in (bx, by, bw, bh))
    return binaria, bbox


def mostrar_segmentacion(ruta_imagen, rect=None, guardar=None, margen=0.05):
    """
    Muestra: original | mascara | objeto segmentado | recorte.
    rect:    (x, y, ancho, alto) opcional para indicar donde esta el objeto.
    guardar: ruta donde guardar el recorte, ej. "imagenes_reales/botella_crop.jpg"
    Devuelve el bounding box (x, y, ancho, alto) del objeto.
    """
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise FileNotFoundError(f"No pude leer la imagen: {ruta_imagen}")

    mascara, (x, y, w, h) = segmentar_objeto(img, rect)

    # Recorte con un pequeno margen alrededor del objeto
    mx, my = int(w * margen), int(h * margen)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, img.shape[1]), min(y + h + my, img.shape[0])
    recorte = img[y0:y1, x0:x1]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aislado = rgb * mascara[:, :, None]

    fig, ejes = plt.subplots(1, 4, figsize=(16, 5))
    paneles = [
        (rgb, "Original"),
        (mascara, "Mascara"),
        (aislado, "Objeto segmentado"),
        (cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB), "Recorte"),
    ]
    for eje, (panel, titulo) in zip(ejes, paneles):
        eje.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        eje.set_title(titulo)
        eje.axis("off")
    plt.tight_layout()
    plt.show()

    if guardar:
        cv2.imwrite(guardar, recorte)
        print(f"Recorte guardado en: {guardar}")
    return (x0, y0, x1 - x0, y1 - y0)
