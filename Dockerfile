# Imagen base con Python 3.11 y pip actualizado
FROM python:3.11-slim

# Evitar prompts interactivos en instalación
ENV DEBIAN_FRONTEND=noninteractive

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements y código
COPY requirements.txt ./
COPY conscious_ai ./conscious_ai
COPY data/training_data.jsonl ./data

 

# Instalar dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y git gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Instalar Python deps
RUN pip install --upgrade pip 
# Instalar Python deps
RUN CMAKE_ARGS="-DGGML_CPU=on -DGGML_NATIVE=OFF" pip install --no-cache-dir -r requirements.txt

# Asegurar que el módulo raíz se detecte
ENV PYTHONPATH="/app"

# Comando por defecto (puedes sobreescribir en docker run)
CMD ["/bin/bash"]
