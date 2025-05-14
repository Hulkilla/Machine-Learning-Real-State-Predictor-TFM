# El código siguiente, que crea un dataframe y quita las filas duplicadas,
# siempre se ejecuta y actúa como un preámbulo del script:

# dataset = pandas.DataFrame()
# dataset = dataset.drop_duplicates()

# Pegue o escriba aquí el código de script:
import pandas as pd
import requests
import matplotlib.pyplot as plt


input_data = {
    "Habitaciones": 3,
    "Aseos": 1,
    "Metros": 105,
    "Comodidades": ["TERRAZA", "PISCINA"],
    "Vivienda": "PISO",
    "CUDIS": 2400802,
}

df = pd.DataFrame(input_data)  # En Power BI es df = dataset.copy()

columnas_necesarias = [
    "CUDIS",
    "Vivienda",
    "Metros",
    "Habitaciones",
    "Aseos",
    "Comodidades",
]

df_filtrado = df[columnas_necesarias]


# Conversiones varias
df_filtrado["Comodidades"] = df_filtrado["Comodidades"].apply(
    lambda x: [c.strip() for c in x.split(",")] if isinstance(x, str) else []
)
df["Vivienda"] = df["Vivienda"].str.upper()


# Mantener "CUDIS" en mayúsculas y convertir las demás columnas a minúsculas
df_filtrado.columns = ["CUDIS"] + [
    col.lower() for col in df_filtrado.columns if col != "CUDIS"
]


# Tomar la primera fila
data_dict = df_filtrado.iloc[0].to_dict()


# URL de la API
url = "https://cr88hyf292.execute-api.eu-west-3.amazonaws.com/v1/predict"
headers = {
    "Content-Type": "application/json",
}


try:
    response = requests.post(
        url,
        json=data_dict,
        headers=headers,
    )

    response.raise_for_status()  # Levanta error para códigos 4xx o 5xx
    r_json = response.json()

    # Intentar extraer la predicción
    prediccion_xgb = float(r_json["predictions"][0]["prediccion_xgb"])

    # Mostrar la predicción como texto
    plt.figure(figsize=(6, 4))
    texto = f"{prediccion_xgb:,.0f}".replace(",", ".") + " €"
    plt.text(
        0.5,
        0.5,
        texto,
        fontsize=60,
        ha='center',
        va='center',
        color="darkslategray",
    )

    plt.axis("off")

except (requests.exceptions.RequestException, KeyError, ValueError) as e:
    # Mostrar mensaje si falla la predicción
    plt.figure(figsize=(10, 4))
    plt.text(
        0.5,
        0.5,
        "No disponible",
        fontsize=60,
        ha='center',
        va='center',
        color="#67002E",
    )
    plt.axis("off")


# Mostrar el gráfico en Power BI
plt.show()
