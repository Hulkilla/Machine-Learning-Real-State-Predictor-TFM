FROM python:3.12-slim

WORKDIR /app

# Copiar código y modelos
COPY app.py .
COPY modelo_ann/ modelo_ann/
COPY modelo_ml/ modelo_ml/
COPY requirements.txt .

# Instalar pip y dependencias de sistema mínimas
RUN  pip install --no-cache-dir -r requirements.txt

# Exponer el puerto de Flask
EXPOSE 8000

# Comando para ejecutar la app
CMD ["python", "app.py"]
