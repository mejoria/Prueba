# -*- coding: utf-8 -*-
import pandas as pd
from google.cloud import bigquery

"""
Módulo para la carga, limpieza y preprocesamiento de datos de ventas.
"""

import pandas as pd
import numpy as np
# Importar otras librerías necesarias a medida que se desarrollen las funciones
# from google.cloud import bigquery

# --- Funciones de Carga ---

def load_data_from_csv(file_path):
    """
    Carga datos desde un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
                      Retorna None si el archivo no se encuentra o hay otro error.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente desde {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {file_path}")
        return None
    except Exception as e:
        print(f"Error inesperado al cargar {file_path}: {e}")
        return None

def load_data_from_bigquery(project_id, dataset_id, table_id):
    """
    Carga datos desde una tabla de BigQuery utilizando google-cloud-bigquery.

    Args:
        project_id (str): ID del proyecto de Google Cloud.
        dataset_id (str): ID del dataset en BigQuery.
        table_id (str): ID de la tabla en BigQuery.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados desde BigQuery.
                      Retorna None si hay errores de conexión o consulta.
    """
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=project_id)
        table_ref = client.dataset(dataset_id).table(table_id)
        table = client.get_table(table_ref)

        df = client.list_rows(table).to_dataframe()
        print(f"Datos cargados exitosamente desde BigQuery: {project_id}.{dataset_id}.{table_id}")
        return df
    except ImportError:
        print("Error: La librería google-cloud-bigquery no está instalada. Por favor, instálala con 'pip install google-cloud-bigquery'")
        return None
    except Exception as e:
        print(f"Error al cargar datos desde BigQuery: {e}")
        return None

# --- Funciones de Limpieza ---

def clean_data(df):
    """
    Realiza la limpieza básica de datos:
        - Imputación de valores nulos (media para numéricas, moda para categóricas).
        - Eliminación de filas duplicadas.

    Args:
        df (pd.DataFrame): DataFrame a limpiar.

    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    print("Iniciando limpieza de datos...")

    # Imputación de valores nulos
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                media = df[col].mean()
                df[col].fillna(media, inplace=True)
                print(f"  Columna '{col}': Nulos imputados con la media ({media:.2f}).")
            else:
                moda = df[col].mode()[0]  # Moda, en caso de múltiples, toma la primera
                df[col].fillna(moda, inplace=True)
                print(f"  Columna '{col}': Nulos imputados con la moda ('{moda}').")

    # Eliminación de duplicados
    n_duplicados_antes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    n_duplicados_despues = df.duplicated().sum()
    if n_duplicados_antes > 0:
        print(f"  Se eliminaron {n_duplicados_antes - n_duplicados_despues} filas duplicadas.")
    else:
        print("  No se encontraron filas duplicadas.")

    print("Limpieza de datos completada.")
    return df

# --- Funciones de Ingeniería de Características ---

def feature_engineering(df):
    """
    Crea nuevas características para el análisis y modelado.
        - Extrae mes y año de la columna 'fecha_venta' (si existe).

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame con nuevas características.
    """
    print("Iniciando ingeniería de características...")

    if 'fecha_venta' in df.columns:
        try:
            df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])  # Asegura que sea datetime
            df['mes_venta'] = df['fecha_venta'].dt.month
            df['anio_venta'] = df['fecha_venta'].dt.year
            print("  Creadas características: 'mes_venta', 'anio_venta' a partir de 'fecha_venta'.")
        except Exception as e:
            print(f"  Error al procesar 'fecha_venta': {e}. Asegúrate de que la columna exista y tenga formato de fecha.")
    else:
        print("  Columna 'fecha_venta' no encontrada. No se crearán características basadas en la fecha.")

    print("Ingeniería de características completada.")
    return df

# --- Funciones de Preprocesamiento ---

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """
    Prepara los datos para el modelado:
        - Limpieza de datos.
        - Ingeniería de características.
        - Codificación de categóricas (One-Hot Encoding).
        - Escalado de numéricas (StandardScaler).
        - División en train/test.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        tuple: (X_train_prep, X_test_prep, y_train, y_test, preprocessor), conjuntos de datos preprocesados y el preprocesador ajustado.
    """
    print("Iniciando preprocesamiento de datos...")

    # 1. Limpieza
    df_cleaned = clean_data(df)

    # 2. Ingeniería de características
    df_featured = feature_engineering(df_cleaned)

    # Separar features (X) y target (y) - Asumiendo 'ventas' como target
    if 'ventas' not in df_featured.columns:
        print("Error: Columna 'ventas' no encontrada. Asegúrate de tener una columna target llamada 'ventas'.")
        return None, None, None, None, None # Retorna None en caso de error
    
    X = df_featured.drop('ventas', axis=1)
    y = df_featured['ventas']

    # 3. Identificar columnas categóricas y numéricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    # 4. Preprocesamiento con ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # 5. Dividir datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Ajustar y transformar preprocesador en datos de entrenamiento y transformar test
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    print("Preprocesamiento de datos completado.")
    return X_train_prep, X_test_prep, y_train, y_test, preprocessor

if __name__ == '__main__':
    # Ejemplo de uso o pruebas rápidas (opcional)
    print("Módulo data_processing cargado.")