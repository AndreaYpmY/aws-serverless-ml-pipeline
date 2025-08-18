import pandas as pd

def test_pandas():
    # Leggo un file CSV
    df = pd.read_csv('../Occupancy_Estimation.csv')

    # Verifico che il DataFrame non sia vuoto
    assert not df.empty, "Il DataFrame è vuoto"

    # Stampo le prime righe del DataFrame
    print(df.head())

    # Info sulle colonne e i tipi di dati
    print(df.info())

    # Stampo le statistiche descrittive
    print(df.describe())

    # Stampo i nomi delle colonne
    print(df.columns.tolist())



    # ==== Filtraggio e Selezione ====

    # Seleziono una colonna specifica
    temperature = df['Temperature']

    # Seleziono più colonne
    temperature_humidity = df[['Temperature', 'Humidity']]

    # Filtraggio basato su una condizione
    high_temperature = df[df['Temperature'] > 25]

    # Filtraggio basato su più condizioni
    high_temp_high_humidity = df[(df['Temperature'] > 25) & (df['Humidity'] > 60)]

    # Ordinamento in base ad una colonna
    sorted_df = df.sort_values(by='Temperature', ascending=False)



    # ==== Modifiche al DataFrame ====

    # Rimuovo righe con valori nulli
    df_cleaned = df.dropna()

    # Sostituisco valori nulli con un valore specifico
    df_filled = df.fillna(0)

    # Rinomino una colonna
    df_renamed = df.rename(columns={'Temperature': 'Temperature_C', 'Humidity': 'Humidity_%'})

    # Aggiungo una nuova colonna 
    df['Temperature_F'] = df['Temperature'] * 9/5 + 32 

    # Salvo il DataFrame modificato in un nuovo file CSV
    df_cleaned.to_csv('../dati_puliti.csv', index=False)




    # Carico solo un numero limitato di righe
    df_sample = pd.read_csv('../Occupancy_Estimation.csv', nrows=10)



if __name__ == "__main__":
    test_pandas()
