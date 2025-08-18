from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

def test_sklearn():
    """
    Esempio di utilizzo di scikit-learn per creare un modello 
    """

    print("Test di scikit-learn per la previsione dell'occupazione...")
    # 1. Carico il file CSV con i dati
    df = pd.read_csv('../Occupancy_Estimation.csv')

    # 2. Divido i dati in "caratteristiche" (X) e "etichetta" (y)
    X = df.drop(columns=['Occupancy'])  # Prendo tutte le colonne tranne 'Occupancy'
    y = df['Occupancy']                 # La colonna da prevedere è 'Occupancy'

    # 3. Divido i dati in due gruppi: addestramento (80%) e test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Normalizzo le caratteristiche: porto i valori su una scala simile
    scaler = StandardScaler()

    # Fit e trasformo i dati di addestramento, poi trasformo quelli di test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. (Opzionale) Codifico variabili categoriche se ce ne sono
    le = LabelEncoder()
    # (Non usato nel codice, ma serve se hai colonne di testo da trasformare in numeri)

    # 6. Creo e addestro un modello di foresta casuale (Random Forest)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 7. Uso il modello per fare previsioni sui dati di test
    y_pred = model.predict(X_test_scaled)

    # 8. Valuto quanto è buono il modello, stampando l’accuratezza
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # 9. Salvo su disco il modello e lo scaler per poterli usare dopo senza riaddestrare
    joblib.dump(model, 'occupancy_model.pkl')
    joblib.dump(scaler, 'occupancy_scaler.pkl')

# 10. Funzione per caricare modello e scaler salvati e fare una predizione su dati nuovi
def load_model_and_scaler(model_path='occupancy_model.pkl', scaler_path='occupancy_scaler.pkl'):
    print("Caricamento del modello e dello scaler...")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Nuovi dati di esempio (con le stesse colonne di X)
    new_data = [[23.5, 45.0, 60.0]]

    # Normalizzo i nuovi dati come fatto prima
    new_data_scaled = scaler.transform(new_data)
    
    # Faccio la predizione con il modello caricato
    prediction = model.predict(new_data_scaled)
    print(f"Predizione per i nuovi dati: {prediction[0]}")



def train():
    print("Avvio il processo di addestramento del modello...")
    # ==== 1. Caricare i dati ====
    df = pd.read_csv('../Occupancy_Estimation.csv')  

    # ==== 2. Selezionare le feature e la variabile target ====
    # Usiamo tutte le colonne numeriche tranne 'Date', 'Time' e 'Room_Occupancy_Count'
    X = df.drop(columns=['Date', 'Time', 'Room_Occupancy_Count'])
    y = df['Room_Occupancy_Count']  # target

    # ==== 3. Dividere i dati in training e test ====
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==== 4. Normalizzare (scalare) i dati ====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==== 5. Addestrare il modello ====
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # ==== 6. Valutare il modello ====
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Accuracy sul test set: {accuracy:.2f}")

    # ==== 7. Salvare modello e scaler ====
    joblib.dump(model, 'tests/files/occupancy_model.pkl')
    joblib.dump(scaler, 'tests/files/occupancy_scaler.pkl')

    print("Modello e scaler salvati nella cartella 'tests/files/'")



# === Funzione quando si prova S3_trigger ===

PROCESSED_DIR = "tests/processed"
MODEL_DIR = "tests/files"

def train_s3_trigger():
    # Ciclo su tutti i CSV preprocessati e li concateno
    all_files = [os.path.join(PROCESSED_DIR, f) for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Feature e target
    X = df.drop(columns=['Room_Occupancy_Count'])
    y = df['Room_Occupancy_Count']

    # Divido i dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardizzo i dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Addestro il modello
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Valuto il modello
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy sul test set: {accuracy:.2f}")

    model_path = os.path.join(MODEL_DIR, "occupancy_model_s3_trigger.pkl")
    scaler_path = os.path.join(MODEL_DIR, "occupancy_scaler_s3_trigger.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modello e scaler salvati nella cartella '{MODEL_DIR}'")








if __name__ == "__main__":   
    #train()
    train_s3_trigger()
    