# Usar una imagen base de Python
FROM python:3.12.6-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos de tu proyecto al contenedor
COPY . /app

#instalar las dependenciad de open cv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000 para que Gunicorn escuche
EXPOSE 8000

# Comando para ejecutar la aplicación con Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
