#### Autorin: Koudjo ####

# Importieren der benötigten Bibliotheken
import streamlit as st
import requests
from urllib.parse import urlparse
#from rmback import isolate_paper  # Annahme: Diese Funktion isoliert das Hauptobjekt im Bild (Hintergrund entfernen)
import cv2

# Flag, um den Zustand der Datei-Uploads und URL-Empfangs zu verfolgen
file_uploaded_and_url_received = False

# Funktion zur Überprüfung des URL-Formats
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Initialisierung des Sitzungszustands für die API-URL, wenn noch nicht festgelegt
if 'API_BASE_URL' not in st.session_state:
    st.session_state['API_BASE_URL'] = 'http://localhost:5000'

# Seitenleiste für die Eingabe der API-URL
with st.sidebar:
    st.title("Pipeline Konfiguration")
    user_api_url = st.text_input("API URL eingeben", value=st.session_state['API_BASE_URL'])
    update_url_button = st.button("URL ändern")

# Aktualisieren der API-Basis-URL im Sitzungszustand, wenn der Button geklickt wird und die URL gültig ist
if update_url_button:
    if is_valid_url(user_api_url):
        st.session_state['API_BASE_URL'] = user_api_url
        st.success("API URL erfolgreich aktualisiert!")
    else:
        st.error("Bitte eine gültige URL eingeben oder die URL ist nicht korrekt.")

# Funktion zum Überprüfen, ob die aktuelle API-URL gültig ist, und Aktivieren/Deaktivieren des Senden-Buttons entsprechend
def is_url_valid():
    return is_valid_url(st.session_state['API_BASE_URL'])

# Hauptlayout und -funktionalität der Seite
st.title('SMARTAPP')

# Funktion zum Senden einer Datei an die API an einem bestimmten Endpunkt
def send_file_to_api(file, endpoint):
    if not is_url_valid():
        st.error("Die API URL ist ungültig. Bitte korrigieren Sie die URL in der Seitenleiste.")
        return
    files = {'files[]': (file.name, file, file.type)}
    full_url = f"{st.session_state['API_BASE_URL']}{endpoint}"
    try:
        response = requests.post(full_url, files=files)
        # Verarbeitung der API-Antwort und Anzeige entsprechender Meldungen
        # (Erfolg, Serverfehler, unerwarteter Fehler)
        # Annahme: Die API antwortet mit JSON-Daten, die 'message' und 'time' enthalten können
        message = response.json().get('message', 'Keine spezifische Nachricht vom Server.')
        time = response.json().get('time', 'Die Laufzeit ist nicht bekannt')
        if response.status_code == 201:
            st.success(message)
            file_uploaded_and_url_received = True
        elif response.status_code == 500:
            st.error(f"Ein Fehler ist auf dem Server aufgetreten: {message}")
        else:
            st.error(f"Ein unerwarteter Fehler ist aufgetreten: {message}")
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Senden der Datei an die API: {e}")

# Funktion zum Initiieren des Herunterladens von Dateien vom Server
def download_files_from_server():
    if not is_url_valid():
        st.error("Die API URL ist ungültig. Bitte korrigieren Sie die URL in der Seitenleiste.")
        return
    full_url = f"{st.session_state['API_BASE_URL']}/downloads"
    try:
        response = requests.get(full_url)
        if response.status_code == 200:
            headers = {
                'Content-Disposition': f'attachment; filename="output_form.pdf"'
            }
            # Streamen des Dateiinhalts als Anhang
            st.write(response.content, headers=headers)
        else:
            st.error("Fehler beim Herunterladen der Dateien.")
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Senden der Anfrage an die API: {e}")

# Datei-Upload-Widget
uploaded_file = st.file_uploader("Wählen Sie ein Bild", type=['jpg', 'jpeg', 'png'])

# Wenn eine Datei hochgeladen wurde, zeige das Bild an
if uploaded_file is not None:
    # Annahme: Die Funktion isolate_paper entfernt den Hintergrund aus dem Bild
    # uploaded_file = isolate_paper(uploaded_file)
    st.image(uploaded_file, caption='Ausgewählte Bild')

# Button zum Auslösen der Bildbearbeitung in der Pipeline
if is_url_valid():
    if st.button('Bearbeitung in pipeline'):
        if uploaded_file is not None:
            send_file_to_api(uploaded_file, '/upload')
        else:
            st.warning("Bitte laden Sie zuerst ein Bild hoch.")
            file_uploaded_and_url_received = False
else:
    st.button('Bearbeitung in pipeline', disabled=True)

# Button zum Herunterladen von Dateien vom Server
if st.button('Datei herunterladen'):
    download_files_from_server()
    # Erstellen des Download-URLs und Anzeige eines Links zum Herunterladen
    download_url = f"{st.session_state['API_BASE_URL']}{'/downloads'}"
    st.write("Klicken Sie unten, um die Datei herunterzuladen:")
    st.markdown(f'<a href="{download_url}" download="output_form.pdf">Herunterladen</a>', unsafe_allow_html=True)
