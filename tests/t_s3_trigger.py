import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import subprocess

# Questo script osserva le cartelle 'input' e 'processed' per nuovi file CSV.
INPUT_DIR = "tests/input"
PROCESSED_DIR = "tests/processed"
PY_PREPROCESS_SCRIPT = "tests/t_preprocess.py"
PY_TRAIN_SCRIPT = "tests/t_sklearn.py"


# Handler per nuovi file in input/
class InputHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            print(f"Nuovo file trovato: {event.src_path}")
            # chiama lo script di preprocessing
            subprocess.run(["python3", PY_PREPROCESS_SCRIPT, event.src_path, PROCESSED_DIR])
            print(f"File preprocessato e salvato in {PROCESSED_DIR}")

# Handler per nuovi file preprocessati
class ProcessedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            print(f"File preprocessato pronto: {event.src_path}")
            # chiama lo script di training
            subprocess.run(["python3", PY_TRAIN_SCRIPT, event.src_path])
            print("Modello addestrato e salvato!")

if __name__ == "__main__":
    observer_input = Observer()
    observer_processed = Observer()

    observer_input.schedule(InputHandler(), INPUT_DIR, recursive=False)
    observer_processed.schedule(ProcessedHandler(), PROCESSED_DIR, recursive=False)

    observer_input.start()
    observer_processed.start()

    print("Watcher avviato. In attesa di nuovi file...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer_input.stop()
        observer_processed.stop()

    observer_input.join()
    observer_processed.join()