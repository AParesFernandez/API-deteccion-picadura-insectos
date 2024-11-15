import os
from dotenv import load_dotenv
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import requests

app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de Roboflow
API_KEY = os.getenv('api_key')
MODEL_ID = os.getenv("model_id")

if not API_KEY or not MODEL_ID:
    raise ValueError("Las variables de entorno 'api_key' y 'model_id' deben estar configuradas correctamente.")

# URL base para la API de Roboflow
ROBOFLOW_URL = f"https://detect.roboflow.com/{MODEL_ID}"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Bearer {API_KEY}"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No se ha subido ninguna imagen."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "message": "No se ha seleccionado ninguna imagen."}), 400

    try:
        # Convertir la imagen usando PIL
        image = Image.open(file)
        # Convertir a base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Hacer la predicción usando requests
        response = requests.post(
            ROBOFLOW_URL,
            data=img_str,
            headers=HEADERS,
        )
        
        if response.status_code == 200:
            result = response.json()
            # Ajusta esto según la estructura de respuesta de tu modelo específico
            prediction = result['predictions'][0]['class'] if result['predictions'] else "No prediction"
            return jsonify({"success": True, "prediction": prediction})
        else:
            return jsonify({"success": False, "message": "Error en la predicción"}), 500
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'success': False, 'message': 'No se recibió imagen.'}), 400

    try:
        # Usar directamente la imagen en base64
        image_data = data['image'].split(',')[1]
        
        # Realizar predicción usando requests
        response = requests.post(
            ROBOFLOW_URL,
            data=image_data,
            headers=HEADERS,
        )
        
        if response.status_code == 200:
            result = response.json()
            # Ajusta esto según la estructura de respuesta de tu modelo específico
            prediction = result['predictions'][0]['class'] if result['predictions'] else "No prediction"
            return jsonify({"success": True, "prediction": prediction})
        else:
            return jsonify({"success": False, "message": "Error en la predicción"}), 500
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)