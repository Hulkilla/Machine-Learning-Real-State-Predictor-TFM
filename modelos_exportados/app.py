from flask import Flask, request, jsonify
from joblib import load
from numpy import log, insert, exp
from pandas import DataFrame
from torch import tensor, float32, no_grad

from modelo_ann.modelo import MLP


# Cargar modelo y escaladores
modelo_ann = load("modelo_ann/modelo_ann.pkl")
scaler_entrada_ann = load("modelo_ann/standardscaler_datos_entrada.pkl")
scaler_precio_ann = load("modelo_ann/standardscaler_precio.pkl")

modelo_xgb = load("modelo_ml/modelo_xgb.pkl")
scaler_entrada_xgb = load("modelo_ml/standardscaler_datos_entrada.pkl")
scaler_precio_xgb = load("modelo_ml/standardscaler_precio.pkl")

# Inicializar la app Flask
app = Flask(__name__)


# Función ANN
def predecir_precio_ann(input_dict, scaler_x, scaler_y, model):
    column_order = ['Habitaciones', 'Aseos', 'Metros', 'CUDIS',
                    'Población', 'Renta bruta media por persona',
                    'Comodidades', 'Capital', 'Precio_medio_mun_tipo']


    input_dict['Precio_medio_mun_tipo'] = log(input_dict['Precio_medio_mun_tipo'])

    features_to_scale = [feat for feat in column_order if feat != 'Capital']
    X_input_to_scale = DataFrame([input_dict], columns=features_to_scale)

    X_scaled = scaler_x.transform(X_input_to_scale)
    capital_index = column_order.index('Capital')
    X_with_capital = insert(X_scaled, capital_index, input_dict['Capital'], axis=1)

    X_tensor = tensor(X_with_capital, dtype=float32)

    model.eval()
    with no_grad():
        y_pred_scaled = model(X_tensor).numpy().ravel()

    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred_euros = exp(y_pred_log)[0]

    return y_pred_euros


# Función XGBoost
def predecir_precio_xgb(input_dict, scaler_x, scaler_y, model):
    column_order = ['Habitaciones', 'Aseos', 'Metros', 'CUDIS', 'Población',
                    'Renta bruta media por persona', 'Comodidades',
                    'Precio_medio_mun_tipo']

    input_dict['Precio_medio_mun_tipo'] = log(input_dict['Precio_medio_mun_tipo'])

    features_to_scale = [feat for feat in column_order if feat != 'Capital']
    X_input_to_scale = DataFrame([input_dict], columns=features_to_scale)

    X_scaled = scaler_x.transform(X_input_to_scale)
    y_pred_scaled = model.predict(X_scaled)

    y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_pred_eur = exp(y_pred_log)

    return y_pred_eur[0][0]


# Endpoint para predicción con XGBoost
@app.route("/predict_xgb", methods=["POST"])
def predict_xgb():
    input_data = request.get_json()
    try:
        pred = predecir_precio_xgb(input_data, scaler_entrada_xgb, scaler_precio_xgb, modelo_xgb)
        return jsonify({"prediccion_xgb": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    

# Endpoint de predicción con ANN
@app.route("/predict_ann", methods=["POST"])
def predict_ann():
    input_data = request.get_json()
    try:
        pred = predecir_precio_ann(input_data, scaler_entrada_ann, scaler_precio_ann, modelo_ann)
        return jsonify({"prediccion_ann": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Test de vida
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
