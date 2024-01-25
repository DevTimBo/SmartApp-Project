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
    identifier_coordinates = {
        "Ausbildungsstätte und Ausbildungsort": (70, 562),
        "Klasse/Fachrichtung": (70, 533),
        "Angestrebter Abschluss": (70, 505),
        "Vollzeitausbildung": (325.5, 487),
        "BAföG-Antrag gestellt": (325.5, 472),
        "Amt für Ausbildungsförderung": (70, 442),
        "Förderungsnummer": (232, 442),
        "Name": (70, 390),
        "Vorname": (70, 362),
        "Geburtsname": (232, 362),
        "Geburtsort": (70, 335),
        "Geschlecht": { "männlich": (292, 335), "weiblich": (232, 335), "divers": (346, 335) },
        "Geburtsdatum": (60, 305),
        "Familienstand": (155, 305),
        "Änderungen ggü. Erklärung": (308, 305),
        "Staatsangehörigkeit": (70, 275),
        "Staatsangehörigkeit Ehegatte/Lebenspartner": (232, 275),
        "Ich habe Kinder": (325, 260),
        "Straße": (70, 207),
        "Hausnummer": (232, 207),
        "Adresszusatz": (290, 207),
        "Land": (57.5, 178),
        "Postleitzahl": (93, 178),
        "Ort": (160, 178),
        "häuslicher Gemeinschaft": (325.5, 132),
        "Eigentum/Mitteleigentum": (325.5, 113),
        "Straße2": (70, 77),
        "Hausnummer2": (232, 77),
        "Adresszusatz2": (290, 77),
        "Land2": (57.5, 47),
        "Postleitzahl2": (93, 47),
        "Ort2": (160, 47),
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
    "Ausbildungsstätte und Ausbildungsort": "Helloo",
    "Klasse/Fachrichtung": "World",
    "Angestrebter Abschluss": "Machine",
    "Vollzeitausbildung": "ja",
    "BAföG-Antrag gestellt": "nein",
    "Amt für Ausbildungsförderung": "Blabla",
    "Förderungsnummer": "1  2  3  4  5  6  7  8  9",
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
    "Straße": "Straße",
    "Hausnummer": "1  5  6  7  8",
    "Adresszusatz": "Adresszusatz",
    "Land": "G  E  R",
    "Postleitzahl": "1  2  3  4  5",
    "Ort": "Ort",
    "häuslicher Gemeinschaft": "ja",
    "Eigentum/Mitteleigentum": "nein",
    "Straße2": "Straße2",
    "Hausnummer2": "1  3  4  5  2",
    "Adresszusatz2": "Adresszusatz2",
    "Land2": "G  E  R",
    "Postleitzahl2": "1  2  3  4  5",
    "Ort2": "Ort2"
}

fill_pdf_form(input_pdf_form, output_pdf_form, input_data)
