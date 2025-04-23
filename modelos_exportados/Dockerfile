FROM python:3.12-slim


WORKDIR /app


# Copiar código y modelos
COPY app.py .
COPY modelo_ann/modelo.py modelo_ann/modelo.py
COPY modelo_ann/modelo_ann.pkl modelo_ann/modelo_ann.pkl
COPY modelo_ml/modelo_xgb.pkl modelo_ml/modelo_xgb.pkl
COPY standardscaler_datos_entrada.pkl .
COPY standardscaler_precio.pkl .
COPY requirements.txt .


# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*


# Paso para evitar errores con compilación de numpy y otros
RUN pip install --upgrade pip setuptools wheel


# Instalar dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt


# Exponer el puerto de Flask
EXPOSE 8000


# Comando para ejecutar la app
CMD ["python", "app.py"]
