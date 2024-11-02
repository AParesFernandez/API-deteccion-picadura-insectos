from flask import Flask

app = Flask(__name__)

# Importa las rutas después de definir la aplicación
import app_flask.routes


