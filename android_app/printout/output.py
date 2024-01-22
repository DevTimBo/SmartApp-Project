import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

def fill_pdf_form(input_pdf, output_pdf, text_input):
    existing_pdf = PyPDF2.PdfReader(open(input_pdf, "rb"))
    output = PyPDF2.PdfWriter()

    # Cause only for the first page
    existing_page = existing_pdf.pages[0]

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    c.setFont("Helvetica", 10)

    for text in text_input:
        word, identifier = text["word"], text["identifier"]

        # Adjusted the coordinates manually cos parsing xml was trash
        #Ausbildung
        if identifier == "Ausbildungsstätte und Ausbildungsort":
            c.drawString(70, 562, f"{word}")
        elif identifier == "Klasse/Fachrichtung":
            c.drawString(70, 533, f"{word}")
        elif identifier == "Angestrebter Abschluss":
            c.drawString(70, 505, f"{word}")
        elif identifier == "Vollzeitausbildung":
            if word == "ja":
                c.drawString(325.5, 487, f"x")
            elif word == "nein":
                c.drawString(363, 487, f"x")
        elif identifier == "BAföG-Antrag gestellt":
            if word == "ja":
                c.drawString(325.5, 472, f"x")
            elif word == "nein":
                c.drawString(363, 472, f"x")
        elif identifier == "Amt für Ausbildungsförderung":
            c.drawString(70, 442, f"{word}")
        elif identifier == "Förderungsnummer":
            c.drawString(232, 442, f"{word}")
        #Angaben zur Person
        elif identifier == "Name":
            c.drawString(70, 390, f"{word}")
        elif identifier == "Vorname":
            c.drawString(70, 362, f"{word}")
        elif identifier == "Geburtsname":
            c.drawString(232, 362, f"{word}")
        elif identifier == "Geburtsort":
            c.drawString(70, 335, f"{word}")
        elif identifier == "Geschlecht":
            if word == "männlich":
                c.drawString(292, 335, f"x")
            elif word == "weiblich":
                c.drawString(232, 335, f"x")
            elif word == "divers":
                c.drawString(346, 335, f"x")
        elif identifier == "Geburtsdatum":
            c.drawString(60, 305, f"{word}")
        elif identifier == "Familienstand":
            c.drawString(155, 305, f"{word}")
        elif identifier == "Änderungen ggü. Erklärung":
            c.drawString(3089, 305, f"{word}")
        elif identifier == "Staatsangehörigkeit":
            c.drawString(70, 275, f"{word}")
        elif identifier == "Staatsangehörigkeit Ehegatte/Lebenspartner":
            c.drawString(232, 275, f"{word}")
        elif identifier == "Ich habe Kinder":
            if word == "ja":
                c.drawString(325, 260, f"x")
            elif word == "nein":
                pass
        #Wohnsitz
        elif identifier == "Straße":
            c.drawString(70, 207, f"{word}")
        elif identifier == "Hausnummer":
            c.drawString(232, 207, f"{word}")
        elif identifier == "Adresszusatz":
            c.drawString(290, 207, f"{word}")
        elif identifier == "Land":
            c.drawString(57.5, 178, f"{word}")
        elif identifier == "Postleitzahl":
            c.drawString(93, 178, f"{word}")
        elif identifier == "Ort":
            c.drawString(160, 178, f"{word}")
        #Ausbildung 2
        elif identifier == "häuslicher Gemeinschaft":
            if word == "ja":
                c.drawString(325.5, 132, f"x")
            elif word == "nein":
                c.drawString(363, 132, f"x")
        elif identifier == "Eigentum/Mitteleigentum":
            if word == "ja":
                c.drawString(325.5, 113, f"x")
            elif word == "nein":
                c.drawString(363, 113, f"x")
        elif identifier == "Straße2":
            c.drawString(70, 77, f"{word}")
        elif identifier == "Hausnummer2":
            c.drawString(232, 77, f"{word}")
        elif identifier == "Adresszusatz2":
            c.drawString(290, 77, f"{word}")
        elif identifier == "Land2":
            c.drawString(57.5, 47, f"{word}")
        elif identifier == "Postleitzahl2":
            c.drawString(93, 47, f"{word}")
        elif identifier == "Ort2":
            c.drawString(160, 47, f"{word}")
        
            
            
    c.save()
    #packet.seek(0) for multiple pages
    
    new_pdf = PyPDF2.PdfWriter()
    new_pdf.add_page(existing_page)

    new_pdf.pages[0].merge_page(PyPDF2.PdfReader(packet).pages[0])
    output.add_page(new_pdf.pages[0])

    with open(output_pdf, "wb") as f:
        output.write(f)

# Example:
input_pdf_form = "android_app/printout/ormblatt_1.pdf"
output_pdf_form = "android_app/printout/output_form.pdf"
text_input = [
    #Ausbildung
    {"word": "Hello", "identifier": "Ausbildungsstätte und Ausbildungsort"},
    {"word": "World", "identifier": "Klasse/Fachrichtung"},
    {"word": "Machine", "identifier": "Angestrebter Abschluss"},
    {"word": "ja", "identifier": "Vollzeitausbildung"},
    {"word": "nein", "identifier": "BAföG-Antrag gestellt"},
    {"word": "Blabla", "identifier": "Amt für Ausbildungsförderung"},
    {"word": "1  2  3  4  5  6  7  8  9", "identifier": "Förderungsnummer"},
    #Angaben zur Person
    {"word": "Name", "identifier": "Name"},
    {"word": "Vorname", "identifier": "Vorname"},
    {"word": "Geburtsname", "identifier": "Geburtsname"},
    {"word": "Geburtsort", "identifier": "Geburtsort"},
    {"word": "männlich", "identifier": "Geschlecht"},
    {"word": "weiblich", "identifier": "Geschlecht"},
    {"word": "divers", "identifier": "Geschlecht"},
    {"word": "0  8  0  9  1  5", "identifier": "Geburtsdatum"},
    {"word": "Familienstand", "identifier": "Familienstand"},
    {"word": "1  2  3  4  5  6", "identifier": "Änderungen ggü. Erklärung"},
    {"word": "Staatsangehörigkeit", "identifier": "Staatsangehörigkeit"},
    {"word": "Staatsangehörigkeit Ehegatte/Lebenspartner", "identifier": "Staatsangehörigkeit Ehegatte/Lebenspartner"},
    {"word": "ja", "identifier": "Ich habe Kinder"},
    #Wohnsitz
    {"word": "Straße", "identifier": "Straße"},
    {"word": "1  5  6  7  8", "identifier": "Hausnummer"},
    {"word": "Adresszusatz", "identifier": "Adresszusatz"},
    {"word": "G  E  R", "identifier": "Land"},
    {"word": "1  2  3  4  5", "identifier": "Postleitzahl"},
    {"word": "Ort", "identifier": "Ort"},
    #Ausbildung 2
    {"word": "ja", "identifier": "häuslicher Gemeinschaft"},
    {"word": "nein", "identifier": "Eigentum/Mitteleigentum"},
    {"word": "Straße2", "identifier": "Straße2"},
    {"word": "1  3  4  5  2", "identifier": "Hausnummer2"},
    {"word": "Adresszusatz2", "identifier": "Adresszusatz2"},
    {"word": "G  E  R", "identifier": "Land2"},
    {"word": "1  2  3  4  5", "identifier": "Postleitzahl2"},
    {"word": "Ort2", "identifier": "Ort2"},
]

fill_pdf_form(input_pdf_form, output_pdf_form, text_input)
