import os
import pandas as pd
from pathlib import Path

# Cartelle di input e output
INPUT_DIR = "tests/input"
PROCESSED_DIR = "tests/processed"

# Creo la cartella processed se non esiste
#Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

# Colonne da tenere per il modello
COLUMNS_TO_KEEP = [
    'S1_Temp','S2_Temp','S3_Temp','S4_Temp',
    'S1_Light','S2_Light','S3_Light','S4_Light',
    'S1_Sound','S2_Sound','S3_Sound','S4_Sound',
    'S5_CO2','S5_CO2_Slope','S6_PIR','S7_PIR',
    'Room_Occupancy_Count'
]

# Ciclo su tutti i CSV nella cartella input
for file_name in os.listdir(INPUT_DIR):
    if file_name.endswith(".csv"):
        file_path = os.path.join(INPUT_DIR, file_name)
        print(f"Preprocessando: {file_path}")

        # Leggo il CSV
        df = pd.read_csv(file_path)

        # Seleziono solo le colonne utili
        df_clean = df[COLUMNS_TO_KEEP]

        # Rimuovo eventuali righe con valori mancanti
        df_clean = df_clean.dropna()

        # Salvo il file preprocessato
        output_path = os.path.join(PROCESSED_DIR, file_name)
        df_clean.to_csv(output_path, index=False)
        print(f"File preprocessato salvato in: {output_path}")
