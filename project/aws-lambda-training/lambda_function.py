# Training_lambda_function
# Funzione AWS Lambda per addestrare un modello di Machine Learning per la stima dell'occupazione di una stanza
# Trigger: Caricamento di un file CSV processato nel bucket S3 'room-occupancy-processed'
# Output: Modello serializzato salvato nel bucket S3 'room-occupancy-models'

import json
import boto3
import pandas as pd
from io import StringIO
import pickle
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    Handler principale della funzione Lambda per l'addestramento del modello.
    Args:
        event: Evento S3 contenente informazioni sul file caricato
        context: Contesto di esecuzione della Lambda
    Returns:
        dict: Risposta JSON con informazioni sul training e metriche
    """
    
    try:
        # Estrazione del bucket e del nome del file dall'evento S3
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        object_key = event['Records'][0]['s3']['object']['key']
        logger.info(f"Avvio addestramento con file: {object_key} dal bucket: {bucket_name}")

        # Controllo che il bucket sia quello atteso
        if bucket_name != 'room-occupancy-processed':
            logger.info(f"Ignoro file dal bucket non valido: {bucket_name}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignorato file dal bucket {bucket_name}'})
            }
        
        # Controllo che il file sia un CSV
        if not object_key.endswith('.csv'):
            logger.info(f"Ignoro file non CSV: {object_key}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignorato file non CSV: {object_key}'})
            }
        
        # Download del file CSV processato da S3
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        # Caricamento del CSV in un DataFrame pandas
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"Dimensione dataset di addestramento: {df.shape}")
        
        # ========== PREPARAZIONE DATI ==========
        
        # Definizione della colonna target
        target_column = 'Room_Occupancy_Count'
        
        # Verifico che la colonna target sia presente nel dataset
        if target_column not in df.columns:
            raise ValueError(f"Colonna target '{target_column}' non trovata nel dataset")
        
        # Separazione delle feature (X) e del target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Feature utilizzate: {list(X.columns)}")
        logger.info(f"Distribuzione del target:\n{y.value_counts().sort_index()}")
        
        # Controllo che ci siano abbastanza dati per l'addestramento
        if len(df) < 20:
            raise ValueError(f"Dati insufficienti per l'addestramento: {len(df)} campioni")
        
        # Divisione in set di addestramento e test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Dimensione set di addestramento: {X_train.shape[0]}")
        logger.info(f"Dimensione set di test: {X_test.shape[0]}")
        
        # ========== ADDESTRAMENTO MODELLO ==========
        
        # Inizializzazione del modello RandomForestClassifier
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        logger.info("Avvio addestramento del modello...")
        
        model.fit(X_train, y_train)
        logger.info("Addestramento completato")
        
        
        # ========== VALUTAZIONE MODELLO ==========
        
        # Predizioni sui set di training e test        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calcolo dell'accuratezza per training e test
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        logger.info("=== VALUTAZIONE MODELLO ===")
        logger.info(f"Accuratezza training: {train_accuracy:.4f}")
        logger.info(f"Accuratezza test: {test_accuracy:.4f}")
        logger.info(f"Report di classificazione (test):\n{classification_report(y_test, y_test_pred)}")
        
        # ========== SALVATAGGIO MODELLO ==========
        
        # Serializzazione del modello 
        model_binary = pickle.dumps(model)
        model_filename = "model_processed.pkl"

        # Salvataggio del modello nel bucket S3
        s3_client.put_object(
            Bucket='room-occupancy-models',
            Key=model_filename,
            Body=model_binary,
            ContentType='application/octet-stream'
        )
        logger.info(f"Modello salvato in s3://room-occupancy-models/{model_filename}")
        
        # ========== RISPOSTA ==========
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Addestramento modello completato con successo',
                'model_file': model_filename,
                'model_performance': {
                    'train_accuracy': float(train_accuracy),
                    'test_accuracy': float(test_accuracy),
                    'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
                },
                'dataset_size': len(df),
                'features_used': list(X.columns)
            })
        }
        
    except Exception as e:
        logger.error(f"Addestramento modello fallito: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Addestramento modello fallito',
                'message': str(e)
            })
        }