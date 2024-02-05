# raspwebapp

## Autorin
Koudjo


## Beschreibung
Die raspwebapp ist eine Anwendung zur Bildbearbeitung, die auf einer API basiert. Sie zeigt die Webanwendung auf dem Port 8501.
Benutzer können Bilder hochladen, diese in der Pipeline bearbeiten lassen und dann die bearbeiteten Dateien herunterladen.
Die Anwendung verwendet die streamlit-Bibliothek für die Benutzeroberfläche und kommuniziert mit einer API, die Bildbearbeitungsdienste bereitstellt.


## Installation

1. Stellen Sie sicher, dass Python auf Ihrem System installiert ist.

2. Installieren Sie die erforderlichen Bibliotheken aus der `requirements.txt`-Datei in einer Umgebung auf dem Pi:

   ```bash
   pip install -r requirements.txt
    ```
3. Anwendung ausführen:

    ```bash
    python raspwebapp.py
    ```


## Usage
1. Open the application in a web browser.
2. Upload images using the provided form.
3. Wait for the image processing to complete.
4. Download the processed images in PDF format.
5. 'PI_API/': Contains image input and output directories, as well as PDF-related folders.

	
## Hinweis
1. Die API-URL kann in der Seitenleiste mithilfe des Textfelds eingegeben und mit dem "URL ändern"-Button aktualisiert werden.
2. Stellen Sie sicher, dass die API ordnungsgemäß konfiguriert ist und auf dem angegebenen Pfad verfügbar ist.
3. Das Hochladen von Bildern und das Auslösen der Bildbearbeitung ist nur möglich, wenn die API-URL gültig ist.
4. Der "Datei herunterladen"-Button ermöglicht das Herunterladen der bearbeiteten Dateien vom Server.
5. Beide Skripte (raspwebapp.py und Smartapp_rasp_api) müssen ausgefüht werden.
