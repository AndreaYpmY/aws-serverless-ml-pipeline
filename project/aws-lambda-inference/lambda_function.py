# Inference_lambda_function
# Funzione AWS Lambda per l'inferenza in tempo reale del modello di stima dell'occupazione di una stanza
# Trigger: Richiesta HTTP tramite API Gateway con dati delle feature
# Output: Predizione e probabilità restituite in formato JSON

import json
import boto3
import pandas as pd
import pickle
import logging
from io import BytesIO


# Configurazione del logging per l'integrazione con AWS CloudWatch
logger = logging.getLogger('inference_lambda')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Inizializzazione del client S3 per interagire con i bucket
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        """
        Handler principale della funzione Lambda per l'inferenza in tempo reale.
        Args:
            event: Evento HTTP contenente il body con i dati delle feature
            context: Contesto di esecuzione della Lambda
        Returns:
            dict: Risposta JSON con predizione, probabilità e messaggio
        """
        logger.info(f"Funzione avviata con RequestId: {context.aws_request_id}")
        logger.info(f"Evento ricevuto: {json.dumps(event, indent=2)}")
        
        # Verifica della presenza del body nella richiesta HTTP
        if 'body' not in event:
            raise ValueError("Nessun body fornito nella richiesta")
        
        body = event['body']
        if isinstance(body, str):
            body = json.loads(body)
        
        # Definizione delle feature attese
        expected_features = [
            'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR'
        ]

        # Validazione delle feature in ingresso
        if not all(feature in body for feature in expected_features):
            missing = [f for f in expected_features if f not in body]
            raise ValueError(f"Feature mancanti: {missing}")
        
        # Creazione di un DataFrame dai dati in ingresso
        data = {k: [body[k]] for k in expected_features}
        df = pd.DataFrame(data)
        logger.info(f"Dimensione dati in ingresso: {df.shape}")
        
        # Calcolo delle feature derivate
        df['Avg_Temperature'] = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)
        df['Avg_Light'] = df[['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']].mean(axis=1)
        df['Total_PIR'] = df[['S6_PIR', 'S7_PIR']].sum(axis=1)
        
        # Caricamento dello scaler da S3
        scaler_key = 'scaler.pkl'  
        response = s3_client.get_object(Bucket='room-occupancy-models', Key=scaler_key)
        scaler = pickle.load(BytesIO(response['Body'].read()))
        logger.info(f"Scaler caricato da s3://room-occupancy-models/{scaler_key}")
        
        # Normalizzazione delle feature numeriche
        features_to_normalize = [
            'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S5_CO2', 'S5_CO2_Slope', 'Avg_Temperature', 'Avg_Light'
        ]
        df[features_to_normalize] = scaler.transform(df[features_to_normalize])
        logger.info("Dati normalizzati")

        # Caricamento del modello da S3
        model_key = 'model_processed.pkl'  
        response = s3_client.get_object(Bucket='room-occupancy-models', Key=model_key)
        model = pickle.load(BytesIO(response['Body'].read()))
        logger.info(f"Modello caricato da s3://room-occupancy-models/{model_key}")
        
        # Esecuzione della predizione
        prediction = model.predict(df)
        probability = model.predict_proba(df).tolist()[0]
        logger.info(f"Predizione: {prediction[0]}, Probabilità: {probability}")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'prediction': int(prediction[0]),
                'probability': probability,
                'message': 'Predizione completata con successo'
            })
        }
    
    except Exception as e:
        logger.error(f"Errore durante l'inferenza: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Inferenza fallita',
                'message': str(e)
            })
        }