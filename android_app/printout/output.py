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

    # Mapping of identifiers to coordinates
    #Names should still be adjusted with input from Hadi
    identifier_coordinates = {
        #Part1
        "Ausbildung_Ausbildungsstätte und Ausbildungsort": (70, 562),
        "Ausbildung_Klasse/Fachrichtung": (70, 533),
        "Ausbildung_Abschluss": (70, 505),
        "Ausbildung_Vollzeit": (325.5, 487),
        "Ausbildung_BAföG-Antrag gestellt": (325.5, 472),
        "Ausbildung_Amt": (70, 442),
        "Ausbildung_Förderungsnummer": (232, 442),
        #Part2
        "Person_Name": (70, 390),
        "Person_Vorname": (70, 362),
        "Person_Geburtsname": (232, 362),
        "Person_Geburtsort": (70, 335),
        "Person_Geschlecht": { "männlich": (292, 335), "weiblich": (232, 335), "divers": (346, 335) },
        "Person_Geburtsdatum": (60, 305),
        "Person_Familienstand": (155, 305),
        "Person_Änderungen ggü. Erklärung": (308, 305),
        "Person_Staatsangehörigkeit": (70, 275),
        "Person_Staatsangehörigkeit_Ehegatte": (232, 275),
        "Person_Kinder": (325, 260),
        #Part3
        "Wohnsitz_Straße": (70, 207),
        "Wohnsitz_Hausnummer": (232, 207),
        "Wohnsitz_Adresszusatz": (290, 207),
        "Wohnsitz_Land": (57.5, 178),
        "Wohnsitz_Postleitzahl": (93, 178),
        "Wohnsitz_Ort": (160, 178),
        #Part4
        "Wohnsitz_waehrend_Ausbildung_häuslicher Gemeinschaft": (325.5, 132),
        "Wohnsitz_waehrend_Ausbildung_Eigentum/Mitteleigentum": (325.5, 113),
        "Wohnsitz_waehrend_Ausbildung_Straße": (70, 77),
        "Wohnsitz_waehrend_Ausbildung_Hausnummer": (232, 77),
        "Wohnsitz_waehrend_Ausbildung_Adresszusatz": (290, 77),
        "Wohnsitz_waehrend_Ausbildung_Land": (57.5, 47),
        "Wohnsitz_waehrend_Ausbildung_Postleitzahl": (93, 47),
        "Wohnsitz_waehrend_Ausbildung_Ort": (160, 47),
    }

    for identifier, word in text_input.items():
        if identifier in identifier_coordinates:
            if isinstance(identifier_coordinates[identifier], tuple):
                c.drawString(*identifier_coordinates[identifier], f"{word}")
            elif isinstance(identifier_coordinates[identifier], dict):
                # Handle gender special case
                for gender, coords in identifier_coordinates[identifier].items():
                    if word == gender:
                        c.drawString(*coords, f"x")
        else:
            print(f"Warning: Identifier {identifier} not found in mapping.")

    c.save()

    new_pdf = PyPDF2.PdfWriter()
    new_pdf.add_page(existing_page)

    new_pdf.pages[0].merge_page(PyPDF2.PdfReader(packet).pages[0])
    output.add_page(new_pdf.pages[0])

    with open(output_pdf, "wb") as f:
        output.write(f)

# Example:
input_pdf_form = "android_app/printout/ormblatt_1.pdf"
output_pdf_form = "android_app/printout/output_form.pdf"
input_data = {
    #Part1
    "Ausbildungsstätte und Ausbildungsort": "Helloo",
    "Klasse/Fachrichtung": "World",
    "Angestrebter Abschluss": "Machine",
    "Vollzeitausbildung": "ja",
    "BAföG-Antrag gestellt": "nein",
    "Amt für Ausbildungsförderung": "Blabla",
    "Förderungsnummer": "1  2  3  4  5  6  7  8  9",
    #Part2
    "Name": "Name",
    "Vorname": "Vorname",
    "Geburtsname": "Geburtsname",
    "Geburtsort": "Geburtsort",
    "Geschlecht": "männlich",
    "Geburtsdatum": "0  8  0  9  1  5",
    "Familienstand": "Familienstand",
    "Änderungen ggü. Erklärung": "1  2  3  4  5  6",
    "Staatsangehörigkeit": "Staatsangehörigkeit",
    "Staatsangehörigkeit Ehegatte/Lebenspartner": "Staatsangehörigkeit Ehegatte/Lebenspartner",
    "Ich habe Kinder": "ja",
    #Part3
    "Wohnsitz_Strasse": "Straße",
    "Wohnsitz_Hausnummer": "1  5  6  7  8",
    "Adresszusatz": "Adresszusatz",
    "Wohnsitz_Land": "G  E  R",
    "Wohnsitz_Postleitzahl": "1  2  3  4  5",
    "Wohnsitz_Ort": "Ort",
    #Part4
    "häuslicher Gemeinschaft": "ja",
    "Eigentum/Mitteleigentum": "nein",
    "Wohnsitz_waehrend_Ausbildung_Strasse": "Straße2",
    "Wohnsitz_waehrend_Ausbildung_Hausnummer": "1  3  4  5  2",
    "Wohnsitz_waehrend_Ausbildung_Adresszusatz": "Adresszusatz2",
    "Wohnsitz_waehrend_Ausbildung_Land": "G  E  R",
    "Wohnsitz_waehrend_Ausbildung_Postleitzahl": "1  2  3  4  5",
    "Wohnsitz_waehrend_Ausbildung_Ort": "Ort2"
}

fill_pdf_form(input_pdf_form, output_pdf_form, input_data)
