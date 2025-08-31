import json
import boto3
import pandas as pd
from io import StringIO
import pickle
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score

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

# Inizializzazione client S3
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """
    Lambda function per il training del modello di occupancy detection.
    Trigger: upload di CSV processato in 'room-occupancy-processed'
    Output: Modello serializzato in 'room-occupancy-models'
    """
    
    try:
        # Estrazione informazioni dal trigger S3
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        object_key = event['Records'][0]['s3']['object']['key']

        logger.info(f"Training model with file: {object_key} from bucket: {bucket_name}")

        # Controlli di base
        if bucket_name != 'room-occupancy-processed':
            logger.info(f"Ignoring file from bucket: {bucket_name}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignored file from {bucket_name}'})
            }
        
        if not object_key.endswith('.csv'):
            logger.info(f"Ignoring non-CSV file: {object_key}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': f'Ignored non-CSV file: {object_key}'})
            }
        

        
        # Download del file CSV processato
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        # Caricamento dati in DataFrame
        df = pd.read_csv(StringIO(csv_content))
        logger.info(f"Training dataset shape: {df.shape}")

        
        # ========== PREPARAZIONE DATI ==========
        
        # Separazione features e target
        target_column = 'Room_Occupancy_Count'
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target distribution: \n{y.value_counts().sort_index()}")
        
        # Controllo dati sufficienti per training
        if len(df) < 20:
            raise ValueError(f"Insufficient data for training: {len(df)} samples")
        
        # Split train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")        
        
        # ========== TRAINING MODELLO ==========
        
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        
        logger.info("Starting model training...")
        
        model.fit(X_train, y_train)
        
        logger.info("Training completed")
        
        
        # ========== VALUTAZIONE MODELLO ==========
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        logger.info("=== MODEL EVALUATION ===")
        logger.info(f"Training MAE: {train_mae:.4f}")
        logger.info(f"Test MAE: {test_mae:.4f}")
        logger.info(f"Training R²: {train_r2:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        
        # ========== SALVATAGGIO MODELLO ==========
        
        model_binary = pickle.dumps(model)
        model_filename = "model_processed.pkl"
        s3_client.put_object(
            Bucket='room-occupancy-models',
            Key=model_filename,
            Body=model_binary,
            ContentType='application/octet-stream'
        )
        logger.info(f"Model saved to s3://room-occupancy-models/{model_filename}")
        
        # ========== RESPONSE ==========
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Model training completed successfully',
                'model_file': model_filename,
                'model_performance': {
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2)
                },
                'dataset_size': len(df),
                'features_used': list(X.columns)
            })
        }
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Model training failed',
                'message': str(e)
            })
        }