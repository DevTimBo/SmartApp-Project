#### Autorin: Koudjo ####

# Importieren der time-Bibliothek für die Zeitmessung
import time

class SimpleTimer:
    def __init__(self):
        # Initialisierung der Timer-Instanz mit Startzeit, Endzeit und bisher verstrichener Zeit
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def start(self):
        # Methode zum Starten des Timers
        self.start_time = time.time()  # Aktuelle Zeit als Startzeit setzen
        self.end_time = None  # Reset der Endzeit jedes Mal, wenn der Timer gestartet wird
        print("Timer started.")

    def stop(self):
        # Methode zum Stoppen des Timers und Berechnen der verstrichenen Zeit
        if self.start_time is None:
            # Wenn der Timer nicht gestartet wurde, wird eine Meldung ausgegeben
            print("Timer was not started!")
            return
        self.end_time = time.time()  # Aktuelle Zeit als Endzeit setzen
        self.elapsed_time = self.end_time - self.start_time  # Berechnen der verstrichenen Zeit
        print("Timer stopped.")

    def get_elapsed_time(self):
        # Methode zum Abrufen der verstrichenen Zeit
        if self.elapsed_time is None:
            # Wenn der Timer noch nicht gestoppt wurde oder nicht gestartet wurde, wird eine Meldung ausgegeben
            print("Timer has not been stopped yet, or has not been started.")
            return 0
        else:
            return self.elapsed_time  # Rückgabe der verstrichenen Zeit in Sekunden

# Beispiel für die Verwendung der SimpleTimer-Klasse:
#timer = SimpleTimer()  # Timer-Instanz erstellen
#timer.start()  # Timer starten
#time.sleep(2)  # Warten für 2 Sekunden (Simulation von Aktivität)
#timer.stop()  # Timer stoppen
#elapsed_time = timer.get_elapsed_time()  # Verstrichene Zeit abrufen
#print(f"Elapsed Time: {elapsed_time} seconds")
#In diesem Beispiel wird ein Timer erstellt, gestartet,
#für 2 Sekunden pausiert (simuliert eine Aktivität) und dann gestoppt.
#Die verstrichene Zeit wird anschließend abgerufen und ausgegeben.