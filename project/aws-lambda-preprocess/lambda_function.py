# Preprocess_lambda_function
# Funzione AWS Lambda per il preprocessing del dataset Room Occupancy Estimation
# Trigger: Caricamento di un file CSV nel bucket S3 'room-occupancy-raw'
# Output: Dataset preprocessato salvato in 'room-occupancy-processed' e scaler in 'room-occupancy-models'

import json
import boto3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import logging
import sys
import pickle

# Configurazione del logging per l'integrazione con AWS CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(levelname)s] %(asctime)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Inizializzazione del client S3 per interagire con i bucket
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Handler principale della funzione Lambda per il preprocessing dei dati.
    Args:
        event: Evento S3 contenente informazioni sul file caricato
        context: Contesto di esecuzione della Lambda
    Returns:
        dict: Risposta JSON con informazioni sul preprocessing e dettagli sul file processato
    """
    try:
        # Estrazione del bucket e del nome del file dall'evento S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        logger.info(f"Elaborazione file: {key} dal bucket: {bucket}")

        # Controllo che il bucket sia quello atteso
        if bucket != 'room-occupancy-raw':
            logger.info(f"Ignoro file dal bucket non valido: {bucket}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignorato file dal bucket {bucket}'})
            }

        # Controllo che il file non sia già stato processato
        if key.startswith('processed_'):
            logger.info(f"Ignoro file già processato: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignorato file già processato: {key}'})
            }
        
        # Controllo che il file sia un CSV
        if key.endswith('.csv') == False:
            logger.info(f"Ignoro file non CSV: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignorato file non CSV: {key}'})
            }
        
        # Lettura del file CSV da S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response['Body'])
        logger.info(f"Dimensione dataset originale: {df.shape}")
        
        # Esecuzione del preprocessing
        df_processed, scaler = preprocess_data(df)
        logger.info(f"Dimensione dataset processato: {df_processed.shape}")
        
        # Definizione dei bucket e chiavi per il salvataggio dei dati processati e dello scaler
        output_bucket = 'room-occupancy-processed'
        output_scaler_bucket = 'room-occupancy-models'
        output_key = f'processed_{key}'
        scaler_key = f'scaler.pkl'
        
        # Salvataggio del dataset processato e dello scaler su S3
        save_to_s3(df_processed, output_bucket, output_key)
        save_scaler_to_s3(scaler, output_scaler_bucket, scaler_key)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'File {key} processato con successo',
                'output_location': f's3://{output_bucket}/{output_key}',
                'original_records': len(df),
                'processed_records': len(df_processed),
                'features': list(df_processed.columns)
            })
        }
    
    except Exception as e:
        logger.error(f"Errore durante l'elaborazione del file: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Errore durante l'elaborazione del file: {str(e)}",
                'file': key if 'key' in locals() else 'unknown'
            })
        }

def preprocess_data(df):
    """
    Esegue il preprocessing del dataset: gestione valori nulli, feature engineering, gestione outlier e normalizzazione.
    Args:
        df: DataFrame pandas contenente il dataset originale
    Returns:
        tuple: DataFrame processato e oggetto scaler StandardScaler
    """

    logger.info("Avvio preprocessing...")
    logger.info(f"Colonne: {list(df.columns)}")
    logger.info(f"Valori mancanti: {df.isnull().sum().sum()}")
    
    # Creazione di una copia del DataFrame per evitare modifiche all'originale
    df_clean = df.copy()
    
    # Gestione valori nulli: rimozione righe con più del 50% di valori mancanti
    null_threshold = len(df_clean.columns) * 0.5
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(thresh=null_threshold)
    logger.info(f"Rimosse {initial_rows - len(df_clean)} righe con >50% valori mancanti")

    # Riempimento dei valori nulli rimanenti con la mediana per le colonne numeriche
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            logger.info(f"Riempiti valori nulli in {col} con mediana: {median_value}")
    

    # Rimozione delle colonne non utili (es. Date, Time)
    columns_to_drop = ['Date', 'Time']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    logger.info(f"Colonne rimosse: {columns_to_drop}")
    
    # Feature engineering: creazione di nuove feature
    # Calcolo della media delle temperature
    temp_cols = [col for col in df_clean.columns if 'Temp' in col]
    if temp_cols:
        df_clean['Avg_Temperature'] = df_clean[temp_cols].mean(axis=1)
        logger.info("Aggiunta feature Avg_Temperature")

    # Calcolo della media della luce
    light_cols = [col for col in df_clean.columns if 'Light' in col]
    if light_cols:
        df_clean['Avg_Light'] = df_clean[light_cols].mean(axis=1)
        logger.info("Aggiunta feature Avg_Light")

    # Calcolo della somma dei sensori PIR
    pir_cols = [col for col in df_clean.columns if 'PIR' in col]
    if pir_cols:
        df_clean['Total_PIR'] = df_clean[pir_cols].sum(axis=1)
        logger.info("Aggiunta feature Total_PIR")
    
    # Gestione degli outlier: clipping basato sull'IQR
    numeric_cols = [col for col in df_clean.select_dtypes(include=['int64', 'float64']).columns
                    if col != 'Room_Occupancy_Count']
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_count = len(df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)])
        if outliers_count > 0:
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            logger.info(f"Clippati {outliers_count} outlier nella colonna {col}")
    
    
    # Normalizzazione delle feature numeriche
    features_to_normalize = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
                            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
                            'S5_CO2', 'S5_CO2_Slope', 'Avg_Temperature', 'Avg_Light']
    features_to_normalize = [col for col in features_to_normalize if col in df_clean.columns]
    if features_to_normalize:
        scaler = StandardScaler()
        df_clean[features_to_normalize] = scaler.fit_transform(df_clean[features_to_normalize])
        logger.info(f"Feature normalizzate: {features_to_normalize}")
    
    # Validazione finale: rimozione di eventuali valori nulli residui
    if df_clean.isnull().sum().sum() > 0:
        logger.warning("Trovati valori NaN dopo il preprocessing")
        df_clean = df_clean.dropna()
        logger.info(f"Rimosse {initial_rows - len(df_clean)} righe con valori NaN residui")

    logger.info(f"Dimensione finale dataset: {df_clean.shape}")
    logger.info(f"Colonne finali: {list(df_clean.columns)}")
    return df_clean, scaler
    
def save_to_s3(df, bucket, key):
    """
    Salva il DataFrame come file CSV su S3.
    Args:
        df: DataFrame da salvare
        bucket: Nome del bucket S3
        key: Chiave del file su S3
    Raises:
        Exception: Se il salvataggio fallisce
    """
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=bucket, 
            Key=key, 
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        logger.info(f"Dati salvati in s3://{bucket}/{key}")
        
    except Exception as e:
        logger.error(f"Errore durante il salvataggio su S3: {str(e)}")
        raise


def save_scaler_to_s3(scaler, bucket, key):
    """
    Salva l'oggetto scaler serializzato su S3.
    Args:
        scaler: Oggetto StandardScaler da salvare
        bucket: Nome del bucket S3
        key: Chiave del file su S3
    Raises:
        Exception: Se il salvataggio fallisce
    """
    try:
        scaler_binary = pickle.dumps(scaler)
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=scaler_binary,
            ContentType='application/octet-stream'
        )
        logger.info(f"Scaler salvato in s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Errore durante il salvataggio dello scaler su S3: {str(e)}")
        raise
