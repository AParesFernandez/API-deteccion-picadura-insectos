import cv2
import os
import numpy as np
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# Obtener la clave API de Roboflow
API_KEY = os.getenv('api_key')

# Inicializar cliente de inferencia de Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key= API_KEY  # Usa la clave API cargada
)

MODEL_ID = os.getenv("model_id")

# Ruta de la imagen local
image_path = "shutterstock_66341713 (350x233).jpg"  # Cambia esto por la ruta de tu imagen

# Leer la imagen usando OpenCV
image = cv2.imread(image_path)

# Verificar si la imagen se ha cargado correctamente
if image is None:
    print(f"Error: No se pudo cargar la imagen en la ruta: {image_path}")
else:
    print("Imagen cargada correctamente.")

    # Enviar la imagen como un array NumPy a Roboflow
    try:
        result = CLIENT.infer(image, model_id="bug-bite-2/1")  # image es np.ndarray
        print(result)  # Mostrar el resultado de la detección
    except Exception as e:
        print(f"Error en la inferencia: {str(e)}")

