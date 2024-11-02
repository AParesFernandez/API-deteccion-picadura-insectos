import base64
from flask import request, jsonify, render_template
from app_flask import app
import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient

# Configuración de Roboflow
API_KEY = os.getenv('api_key')
MODEL_ID = os.getenv("model_id")

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key=API_KEY
)

@app.route('/')
def home():
    # Renderiza la página principal con la interfaz de la cámara
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No se ha subido ninguna imagen."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "message": "No se ha seleccionado ninguna imagen."}), 400

    try:
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "message": "Error al cargar la imagen."}), 400

        # Aquí haz la predicción con el modelo
        result = CLIENT.infer(image, model_id=MODEL_ID)  # Asegúrate de definir CLIENT y MODEL_ID correctamente
        return jsonify({"success": True, "prediction": result})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    
@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    data = request.get_json()  # Obtener datos JSON
    if 'image' not in data:
        return jsonify({'success': False, 'message': 'No se recibió imagen.'}), 400

    # Decodificar la imagen de base64
    image_data = data['image'].split(',')[1]  # Eliminar la parte 'data:image/png;base64,'
    image_data = base64.b64decode(image_data)  # Decodificar base64
    np_image = np.frombuffer(image_data, np.uint8)  # Convertir a array de NumPy
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)  # Leer la imagen con OpenCV

    # Realizar inferencia
    try:
        result = CLIENT.infer(image, model_id=MODEL_ID)
        return jsonify({'success': True, 'prediction': result})  # Retornar la predicción
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
