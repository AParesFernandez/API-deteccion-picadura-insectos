from flask import Flask
from dotenv import load_dotenv
import os

from matplotlib import pyplot as plt

# Cargar las variables de entorno
load_dotenv()

app = Flask(__name__)

# Cargar las rutas desde routes.py
with app.app_context():
    import routes  # Esto registra las rutas en la aplicación Flask

if __name__ == '__main__':
    plt.show()

