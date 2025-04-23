import requests

input_data = {
    "Habitaciones": 2,
    "Aseos": 3,
    "Metros": 85,
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

