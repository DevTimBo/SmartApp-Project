# Bildverarbeitungsanwendung mit Flask und PiCamera2

## Autorin
Koudjo


## Beschreibung
Diese Anwendung ermöglicht das Erfassen von Bildern über eine Raspberry Pi-Kamera, isoliert das darauf befindliche Papier und wendet eine perspektivische Transformation an. Die Anwendung ist mit Flask implementiert und verwendet die PiCamera2-Bibliothek für die Kamerafunktionalität.
Hier ist eine Übersicht über die Funktionalitäten:

1. Es importiert notwendige Module wie Flask für die Webanwendung, die PiCamera2-Bibliothek für die Kamerafunktionalität, OpenCV für Bildverarbeitungsaufgaben und NumPy für numerische Berechnungen.

2. Es erstellt eine Flask-App und definiert einen Ordner, in dem hochgeladene Bilder gespeichert werden sollen.

3. Es definiert Funktionen zur Bestimmung der Reihenfolge der Punkte eines Rechtecks und zur perspektivischen Transformation eines Bildes.

4. Die Flask-App definiert eine Route "/capture", die beim Aufrufen ein Bild mit der angeschlossenen Pi-Kamera erfasst. Das erfasste Bild wird dann isoliert und perspektivisch transformiert.

5. Die `isolate_paper`-Funktion verwendet Bildverarbeitungstechniken wie Graustufenkonvertierung, Gaußsche Unschärfe und Kantenerkennung, um das Papier in einem Bild zu isolieren.

6. Die `capture_image`-Funktion startet die Pi-Kamera, erfasst ein Bild, isoliert das Papier und speichert das transformierte Bild als "transformed.jpg".

7. Die Flask-App wird gestartet und läuft auf dem Host '0.0.0.0' und Port 5000 im Debug-Modus, sodass Fehlermeldungen und Debugging-Informationen angezeigt werden können.


## Installation
1. Installiere die erforderlichen Pakete aus der `requirements.txt`-Datei mit dem Befehl in einer Umgebung auf dem Pi::
    ```bash
    pip install -r requirements.txt
    ```
2. Führe die Flask-App aus:

    ```bash
    python app.py
    ```

Die Anwendung wird auf `http://0.0.0.0:5000/` gestartet.

## Funktionen

- **/capture:** Erfasst ein Bild mit der Pi-Kamera, isoliert das Papier in dem Bild und wendet eine perspektivische Transformation an. Das erfasste Bild wird als "test.jpg" gespeichert, und das transformierte Bild wird als "transformed.jpg" gespeichert.


## Hinweis
Stelle sicher, dass die Raspberry Pi-Kamera ordnungsgemäß angeschlossen ist.