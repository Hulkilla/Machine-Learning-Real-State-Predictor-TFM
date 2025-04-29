import requests

# Prueba de la API local con el modelo de XGBoost y ANN exportados
# Se asume que el servidor FastAPI está corriendo en localhost:8000
input_data = {
    "Habitaciones": 3,
    "Aseos": 2,
    "Metros": 105,
    "CUDIS": 2408907,
    "Población": 120000,
    "Renta bruta media por persona": 36000,
    "Comodidades": 1,
    "Capital": 1,
    "Precio_medio_mun_tipo": 132000
}

r = requests.get("http://localhost:8000/health", json=input_data)
print(r.json())

r = requests.post("http://localhost:8000/predict_xgb", json=input_data)
print(r.json())

r = requests.post("http://localhost:8000/predict_ann", json=input_data)
print(r.json())


# Prueba de la API en AWS para mandar los datos desde el Power BI
# Se asume que el servidor FastAPI está corriendo en AWS
input_data = {
    "habitaciones": 3,
    "aseos": 1,
    "metros": 105,
    "comodidades": ["TERRAZA","PISCINA"],
    "vivienda": "PISO",
    "CUDIS": 2400802
}

r = requests.post("https://cr88hyf292.execute-api.eu-west-3.amazonaws.com/v1/predict", json=input_data)
r_json = r.json()
print(r_json)
# Obtener el valor de la predicción XGB
prediccion_xgb = float(r_json["predictions"][0]["prediccion_xgb"])
print("Predicción XGB:", prediccion_xgb)


import matplotlib.pyplot as plt
# Crear el gráfico de barras
plt.figure(figsize=(6,4))
plt.bar(["Predicción XGB"], [prediccion_xgb], color="royalblue")
plt.ylabel("Valor de predicción")
plt.title("Resultado de la predicción XGB")

# Mostrar el gráfico en Power BI
plt.savefig("imagenes/prediccion_xgb.png")
