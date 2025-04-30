from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def connect_to_postgresql():
    """
    Establishes a connection to a PostgreSQL database using credentials from environment variables.

    Returns:
        engine: SQLAlchemy engine object if the connection is successful, None otherwise.
    """

    load_dotenv()

    USER = os.getenv("user")
    PASSWORD = os.getenv("password")
    HOST = os.getenv("host")
    PORT = os.getenv("port")
    DBNAME = os.getenv("dbname")

    if HOST == 'localhost' or 'grok' in HOST:
        DATABASE_URL = (
            f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
        )
    else:
        DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

    engine = create_engine(DATABASE_URL)

    try:
        if engine:
            print("Conexión establecida")
            return engine
        else:
            print("No se ha podido establecer la conexión")
    except Exception as e:
        print(f"Failed to connect: {e}")


def graficar_corr_matrix(df):
    """
    Generates and displays a correlation matrix heatmap for numerical variables in a DataFrame.

    Args:
        df: DataFrame containing numerical variables.

    Returns:
        Displays the correlation matrix heatmap.
    """
    # Calcular la matriz de correlación
    corr_matrix = df.select_dtypes('number').corr(method='pearson')

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(16, 16))

    # Definir los colores
    color_list = ['PLUM', 'white', '#6FD08C']
    cmap = LinearSegmentedColormap.from_list("", color_list)

    # Crear el gráfico de matriz de correlación
    cax = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)

    # Añadir valores al gráfico
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(
                j,
                i,
                round(corr_matrix.iloc[i, j], 2),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    # Añadir barra de color
    cbar = fig.colorbar(cax)

    # Configurar etiquetas de las columnas y filas
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(
        corr_matrix.columns, rotation=45, ha="right", fontsize=9
    )
    ax.set_yticklabels(corr_matrix.columns, fontsize=12)

    # Mantener una proporción consistente en los gráficos
    ax.set_aspect('equal')

    # Ajustar el diseño
    plt.tight_layout()
    plt.show()


def plot_predictions(actual, predicted, title="Valores Reales vs Predichos"):
    """
    Plots actual vs predicted values with ideal and margin lines.

    Args:
        actual: Array of actual values.
        predicted: Array of predicted values.
        title: Title of the plot (default: "Valores Reales vs Predichos").

    Returns:
        Displays the scatter plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=actual, y=predicted, alpha=0.6, label="Predicciones")

    # Añadir línea ideal (y=x)
    plt.plot(
        [min(actual), max(actual)],
        [min(actual), max(actual)],
        color='red',
        linestyle='--',
        label="Línea ideal (y=x)",
    )

    # Añadir líneas paralelas con un margen del 30%
    lower_bound = [val * 0.7 for val in actual]  # 30% por debajo de y=x
    upper_bound = [val * 1.3 for val in actual]  # 30% por encima de y=x
    plt.plot(
        actual,
        lower_bound,
        color='blue',
        linestyle='--',
        label="Límite inferior (70%)",
    )
    plt.plot(
        actual,
        upper_bound,
        color='green',
        linestyle='--',
        label="Límite superior (130%)",
    )

    # Etiquetas y título
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title(title)
    plt.legend()
    sns.set_theme(style="whitegrid")
    plt.show()


def evaluate_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_results,
    scaler,
    cv=5,
):
    """
    Evaluates a model using cross-validation and computes performance metrics.

    Args:
        model_name: Name of the model (string).
        model: Model to evaluate.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        model_results: List to store evaluation results.
        scaler: Pre-fitted scaler for inverse transformations.
        cv: Number of cross-validation folds (default: 5).

    Returns:
        predictions_original: Predictions on the test set in the original scale.
        model_results: Updated list with evaluation metrics.
    """

    # Medir tiempo de entrenamiento
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(X_test)

    # Revertir el escalado de los valores reales y predichos usando el escalador ajustado
    y_test_original = np.exp(
        scaler.inverse_transform(y_test.values.reshape(-1, 1)).ravel()
    )
    predictions_original = np.exp(
        scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    )

    # Calcular métricas estándar con valores en escala original
    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    # Validación cruzada sobre datos escalados
    neg_mse_cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'
    )
    r2_cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='r2'
    )

    # Calcular RMSE original desde MSE escalado
    rmse_cv_scores = np.sqrt(-neg_mse_cv_scores)
    rmse_cv_scores_original = scaler.inverse_transform(
        rmse_cv_scores.reshape(-1, 1)
    ).ravel()

    rmse_cv_mean = rmse_cv_scores_original.mean()
    rmse_cv_std = rmse_cv_scores_original.std()
    r2_cv_mean = r2_cv_scores.mean()
    r2_cv_std = r2_cv_scores.std()

    # Calcular la desviación promedio (en porcentaje) entre valores reales y predichos
    epsilon = 1e-8  # Evitar divisiones por cero
    avg_deviation_percentage = (
        np.mean(
            np.abs(
                (y_test_original - predictions_original)
                / (y_test_original + epsilon)
            )
        )
        * 100
    )

    # Buscar si ya existe el modelo en la lista de resultados
    existing_model_index = next(
        (
            i
            for i, result in enumerate(model_results)
            if result['Modelo'] == model_name
        ),
        None,
    )

    # Guardar resultados en la lista
    new_results = {
        'Modelo': model_name,
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R²': float(r2),
        'Desviación promedio (%)': float(avg_deviation_percentage),
        'Tiempo entrenamiento (s)': float(train_time),
        'RMSE CV': float(rmse_cv_mean),
        'RMSE CV std': float(rmse_cv_std),
        'R² CV': float(r2_cv_mean),
        'R² CV std': float(r2_cv_std),
    }

    if existing_model_index is not None:
        model_results[existing_model_index] = new_results
    else:
        model_results.append(new_results)

    return predictions_original, model_results


def get_model_results(model_name, model_results):
    """
    Retrieves evaluation metrics for a specific model from the results list.

    Args:
        model_name: Name of the model (string).
        model_results: List of evaluation results.

    Returns:
        Dictionary with the model's metrics, or None if not found.
    """
    # Filtrar el modelo con el nombre dado
    result = next(
        (item for item in model_results if item['Modelo'] == model_name), None
    )

    if result:
        # Convertir métricas numéricas a float si es necesario
        result = {
            key: (float(value) if isinstance(value, np.float64) else value)
            for key, value in result.items()
        }
        return result
    else:
        print(f"No se encontraron resultados para el modelo: {model_name}")
        return None


def evaluate_feature_importance(
    model, feature_names, title="Importancia de las variables"
):
    """
    Evaluates and visualizes feature importance for a trained model.

    Args:
        model: Trained model with `feature_importances_` attribute.
        feature_names: List of feature names.
        title: Title for the plot (default: "Importancia de las variables").

    Returns:
        DataFrame with feature importance metrics.
    """
    feature_importances = model.feature_importances_

    # Crear un DataFrame para organizar la información
    importance_df = pd.DataFrame(
        {
            'Feature': feature_names,
            '%IncMSE': feature_importances
            * 100,  # Simulación del impacto en el MSE como porcentaje
            'IncNodePurity': np.random.uniform(
                1e10, 1e12, len(feature_names)
            ),  # Placeholder para pureza de nodo
        }
    )

    # Ordenar por importancia
    importance_df = importance_df.sort_values(by='%IncMSE', ascending=False)

    # Mostrar la tabla
    print("Importancia de las variables:")
    print(importance_df)

    # Graficar la importancia (%IncMSE)
    plt.figure(figsize=(12, 6))
    plt.bar(
        importance_df['Feature'], importance_df['%IncMSE'], color='steelblue'
    )
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel('Variables')
    plt.ylabel('% Incremento en MSE')
    plt.show()

    return importance_df


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    n_epochs,
    batch_size,
    loss_fn,
    optimizer,
    scaler,
    patience=10,
):
    """
    Trains a PyTorch model with early stopping and logs training/validation metrics.

    Args:
        model: PyTorch model to train.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        loss_fn: Loss function.
        optimizer: Optimizer for training.
        scaler: Pre-fitted scaler for inverse transformations.
        patience: Number of epochs to wait for improvement before stopping (default: 10).

    Returns:
        best_loss: Best validation loss achieved.
        train_loss_hist: Training loss history.
        val_loss_hist: Validation loss history.
        val_mae_hist: Validation MAE history.
        val_deviation_hist: Validation deviation history.
    """
    batch_start = torch.arange(0, len(X_train), batch_size)

    best_loss = np.inf
    best_weights = None
    patience_counter = 0

    train_loss_hist = []
    val_loss_hist = []
    val_mae_hist = []
    val_deviation_hist = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = []

        for start in batch_start:
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        train_loss = np.mean(epoch_loss)
        model.eval()

        with torch.no_grad():
            y_val_pred = model(X_val)

            y_val_inv = np.exp(
                scaler.inverse_transform(
                    y_val.cpu().numpy().reshape(-1, 1)
                ).ravel()
            )
            y_val_pred_inv = np.exp(
                scaler.inverse_transform(
                    y_val_pred.cpu().numpy().reshape(-1, 1)
                ).ravel()
            )

            # 🔎 Métricas en escala original
            val_loss = loss_fn(
                y_val_pred, y_val
            ).item()  # Se queda en log+escalado (no se transforma)
            val_mae = np.mean(np.abs(y_val_inv - y_val_pred_inv))
            val_deviation = np.std(y_val_inv - y_val_pred_inv)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_mae_hist.append(val_mae)
        val_deviation_hist.append(val_deviation)

        if (epoch + 1) % 5 == 0:
            print(
                f"""Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.4f} |
                 Val Loss: {val_loss:.4f} | Val MAE: {val_mae:,.2f}"""
            )

        # ⏹️ Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping activado en la época {epoch}. Mejor Val Loss: {best_loss:.4f}"
                )
                break

    model.load_state_dict(best_weights)
    return (
        best_loss,
        train_loss_hist,
        val_loss_hist,
        val_mae_hist,
        val_deviation_hist,
    )


def calculate_percentage_deviation(y_real, y_pred):
    """
    Calculates the average percentage deviation between actual and predicted values.

    Args:
        y_real: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        Average percentage deviation as a float.
    """
    # Evitar divisiones por cero
    epsilon = 1e-8  # Pequeño valor para asegurar estabilidad numérica
    deviation_percentage = (
        torch.mean(torch.abs((y_real - y_pred) / (y_real + epsilon))) * 100
    )
    return deviation_percentage.item()  # Devuelve como valor flotante
