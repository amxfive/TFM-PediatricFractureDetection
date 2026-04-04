FROM python:3.10-slim

# 1. Dependencias de sistema para OpenCV y utilidades
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalación de librerías asegurando el PATH
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copiar el código y el modelo
COPY . .

# 4. TRUCO DE ORO: Usar la ruta absoluta del módulo Python
# Esto evita que dependamos de si 'uvicorn' está en el PATH o no
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]