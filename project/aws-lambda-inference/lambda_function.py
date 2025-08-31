import json
import boto3
import pandas as pd
import pickle
import logging
from io import BytesIO

logger = logging.getLogger('inference_lambda')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        logger.info(f"Function started with RequestId: {context.aws_request_id}")
        logger.info(f"Event received: {json.dumps(event, indent=2)}")
        
        # Estraggo il body della richiesta HTTP
        if 'body' not in event:
            raise ValueError("No body provided in the request")
        
        body = event['body']
        if isinstance(body, str):
            body = json.loads(body)
        
        # Valido i dati in ingresso
        expected_features = [
            'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR'
        ]
        if not all(feature in body for feature in expected_features):
            missing = [f for f in expected_features if f not in body]
            raise ValueError(f"Missing features: {missing}")
        
        # Creo DataFrame dai dati in ingresso
        data = {k: [body[k]] for k in expected_features}
        df = pd.DataFrame(data)
        logger.info(f"Input data shape: {df.shape}")
        
        # Calcolo feature derivate
        df['Avg_Temperature'] = df[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)
        df['Avg_Light'] = df[['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']].mean(axis=1)
        df['Total_PIR'] = df[['S6_PIR', 'S7_PIR']].sum(axis=1)
        
        # Carico lo scaler da S3
        scaler_key = 'scaler.pkl'  
        response = s3_client.get_object(Bucket='room-occupancy-models', Key=scaler_key)
        scaler = pickle.load(BytesIO(response['Body'].read()))
        logger.info(f"Scaler loaded from s3://room-occupancy-models/{scaler_key}")
        
        # Normalizzo i dati
        features_to_normalize = [
            'S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S5_CO2', 'S5_CO2_Slope', 'Avg_Temperature', 'Avg_Light'
        ]
        df[features_to_normalize] = scaler.transform(df[features_to_normalize])
        logger.info("Data normalized")
        features = expected_features + ['Avg_Temperature', 'Avg_Light', 'Total_PIR']

        
        # Carico il modello da S3
        model_key = 'model_processed.pkl'  
        response = s3_client.get_object(Bucket='room-occupancy-models', Key=model_key)
        model = pickle.load(BytesIO(response['Body'].read()))
        logger.info(f"Model loaded from s3://room-occupancy-models/{model_key}")
        
        # Eseguo la predizione
        prediction = model.predict(df)
        probability = model.predict_proba(df).tolist()[0]
        logger.info(f"Prediction: {prediction[0]}, Probability: {probability}")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'prediction': int(prediction[0]),
                'probability': probability,
                'message': 'Prediction completed successfully'
            })
        }
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Inference failed',
                'message': str(e)
            })
        }