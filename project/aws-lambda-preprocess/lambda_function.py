import json
import boto3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import logging
import sys
import pickle

# Configurazione del logging per Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Configura handler per CloudWatch
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(levelname)s] %(asctime)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Inizializzo il client S3
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Funzione Lambda per il preprocessing dei dati.
    Trigger: upload di CSV in 'room-occupancy-raw'
    Output: CSV processato in 'room-occupancy-processed'
    """
    try:
        # Estraggo bucket e chiave del file
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Processing file: {key} from bucket: {bucket}")

        # Alcuni controlli di base
        if bucket != 'room-occupancy-raw':
            logger.info(f"Ignoring file from bucket: {bucket}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignored file from {bucket}'})
            }
    
        if key.startswith('processed_'):
            logger.info(f"Ignoring already processed file: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignored already processed file: {key}'})
            }
        
        if key.endswith('.csv') == False:
            logger.info(f"Ignoring non-CSV file: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignored non-CSV file: {key}'})
            }
        
        # Leggo il file CSV da S3
        response = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response['Body'])
        
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Preprocessing
        df_processed, scaler = preprocess_data(df)
        
        logger.info(f"Processed dataset shape: {df_processed.shape}")
        
        # Salvo il file preprocessato nel bucket room-occupancy-processed e scaler in room-occupancy-models
        output_bucket = 'room-occupancy-processed'
        output_scaler_bucket = 'room-occupancy-models'
        output_key = f'processed_{key}'
        scaler_key = f'scaler.pkl'

        
        # Verifico che il bucket esista
        #ensure_bucket_exists(output_bucket)
        
        # Salvataggio 
        save_to_s3(df_processed, output_bucket, output_key)
        save_scaler_to_s3(scaler, output_scaler_bucket, scaler_key)

        
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'File {key} processed successfully',
                'output_location': f's3://{output_bucket}/{output_key}',
                'original_records': len(df),
                'processed_records': len(df_processed),
                'features': list(df_processed.columns)
            })
        }
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f"Error processing file: {str(e)}",
                'file': key if 'key' in locals() else 'unknown'
            })
        }

def preprocess_data(df):
    
    # 1. Controllo iniziale del dataset
    logger.info("Starting preprocessing...")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    
    # Copio per evitare modifiche al dataframe originale
    df_clean = df.copy()
    
    # 2. Gestione valori nulli (più sofisticata)
    # Rimuovo righe con più del 50% di valori nulli
    null_threshold = len(df_clean.columns) * 0.5
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(thresh=null_threshold)
    
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with >50% missing values")

    # Riempio valori nulli rimanenti con la mediana per colonne numeriche
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().any():
            
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            
            logger.info(f"Filled {col} nulls with median: {median_value}")
    

    # 3. Rimozione colonne non utili
    columns_to_drop = ['Date', 'Time']
    df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
    logger.info(f"Dropped columns: {columns_to_drop}")
    
    # 4. Feature engineering
    # Media delle temperature
    temp_cols = [col for col in df_clean.columns if 'Temp' in col]
    if temp_cols:
        df_clean['Avg_Temperature'] = df_clean[temp_cols].mean(axis=1)
        logger.info("Added Avg_Temperature feature")

    # Media della luce
    light_cols = [col for col in df_clean.columns if 'Light' in col]
    if light_cols:
        df_clean['Avg_Light'] = df_clean[light_cols].mean(axis=1)
        logger.info("Added Avg_Light feature")

    # Somma dei sensori PIR
    pir_cols = [col for col in df_clean.columns if 'PIR' in col]
    if pir_cols:
        df_clean['Total_PIR'] = df_clean[pir_cols].sum(axis=1)
        logger.info("Added Total_PIR feature")
    
    # 5. Gestione degli outlier (clipping invece di rimozione)
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
            logger.info(f"Clipped {outliers_count} outliers in column {col}")
    
    
    # 6. Normalizzazione delle feature numeriche
    features_to_normalize = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
                            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
                            'S5_CO2', 'S5_CO2_Slope', 'Avg_Temperature', 'Avg_Light']
    features_to_normalize = [col for col in features_to_normalize if col in df_clean.columns]
    if features_to_normalize:
        scaler = StandardScaler()
        df_clean[features_to_normalize] = scaler.fit_transform(df_clean[features_to_normalize])
        logger.info(f"Normalized features: {features_to_normalize}")
    
     # 6. Validazione finale
    if df_clean.isnull().sum().sum() > 0:
        logger.warning("Still some NaN values after preprocessing")
        df_clean = df_clean.dropna()
        logger.info(f"Removed {initial_rows - len(df_clean)} rows with remaining NaN values")

    logger.info(f"Final dataset shape: {df_clean.shape}")
    logger.info(f"Final columns: {list(df_clean.columns)}")
    return df_clean, scaler
    



def ensure_bucket_exists(bucket_name):
    """Verifico che il bucket esista, altrimenti errore."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} exists")
    except Exception as e:
        logger.warning(f"Cannot access bucket {bucket_name}: {str(e)}")
        raise Exception(f"Bucket {bucket_name} not accessible")



def save_to_s3(df, bucket, key):
    """Salvataggio ottimizzato su S3"""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=bucket, 
            Key=key, 
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        logger.info(f"Data saved to s3://{bucket}/{key}")
        
    except Exception as e:
        logger.error(f"Error saving to S3: {str(e)}")
        raise


def save_scaler_to_s3(scaler, bucket, key):
    """Salvataggio dello scaler su S3"""
    try:
        scaler_binary = pickle.dumps(scaler)
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=scaler_binary,
            ContentType='application/octet-stream'
        )
        logger.info(f"Scaler saved to s3://{bucket}/{key}")
        
    except Exception as e:
        logger.error(f"Error saving scaler to S3: {str(e)}")
        raise
