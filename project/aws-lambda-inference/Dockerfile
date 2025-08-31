FROM public.ecr.aws/lambda/python:3.11

# Copia il codice della funzione
COPY lambda_function.py .

# Installa le dipendenze
RUN pip install --no-cache-dir pandas==1.5.3 scikit-learn==1.2.2 numpy==1.26.4

# Specifica il handler
CMD ["lambda_function.lambda_handler"]