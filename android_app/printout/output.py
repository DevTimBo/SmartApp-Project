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
        #Part1
        "Ausbildung_Staette": (70, 562),
        "Ausbildung_Klasse": (70, 533),
        "Ausbilung_Abschluss": (70, 505),
        "Ausbildung_Vollzeit": (325.5, 487),
        "Ausbildung_Teilzeit": (363, 487),
        "Ausbildung_Antrag_gestellt_ja": (325.5, 472),
        "Ausbildung_Antrag_gestellt_nein": (363, 472),
        "Ausbildung_Amt": (70, 442),
        "Ausbildung_Foerderungsnummer": (231, 442),
        #Part2
        "Person_Name": (70, 390),
        "Person_Vorname": (70, 362),
        "Person_Geburtsname": (232, 362),
        "Person_Geburtsort": (70, 335),
        "Person_maennlich": (292, 335), 
        "Person_weiblich": (232, 335), 
        "Person_divers": (346, 335),
        "Person_Geburtsdatum": (59, 305),
        "Person_Familienstand": (155, 305),
        "Person_Familienstand_seit": (308, 305),
        "Person_Stattsangehörigkeit_eigene": (70, 275),
        "Person_Stattsangehörigkeit_Ehegatte": (232, 275),
        "Person_Kinder": (325, 260),
        #Part3
        "Wohnsitz_Strasse": (70, 207),
        "Wohnsitz_Hausnummer": (232, 207),
        "Wohnsitz_Adresszusatz": (290, 207),
        "Wohnsitz_Land": (57.5, 178),
        "Wohnsitz_Postleitzahl": (93, 178),
        "Wohnsitz_Ort": (160, 178),
        #Part4
        "Wohnsitz_waehrend_Ausbildung_elternmiete": (325.5, 132),
        "Wohnsitz_waehrend_Ausbildung_elternmiete_nein": (363, 132),
        "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja": (325.5, 113),
        "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein": (363, 113),
        "Wohnsitz_waehrend_Ausbildung_Strasse": (70, 77),
        "Wohnsitz_waehrend_Ausbildung_Hausnummer": (232, 77),
        "Wohnsitz_waehrend_Ausbildung_Adresszusatz": (290, 77),
        "Wohnsitz_waehrend_Ausbildung_Land": (57.5, 47),
        "Wohnsitz_waehrend_Ausbildung_Postleitzahl": (93, 47),
        "Wohnsitz_waehrend_Ausbildung_Ort": (160, 47),
    }
    
    special_spacing_identifiers = ["Ausbildung_Foerderungsnummer", "Person_Geburtsdatum", "Person_Familienstand_seit",
                                   "Wohnsitz_Hausnummer", "Wohnsitz_Land", "Wohnsitz_Postleitzahl",
                                   "Wohnsitz_waehrend_Ausbildung_Hausnummer", "Wohnsitz_waehrend_Ausbildung_Land", "Wohnsitz_waehrend_Ausbildung_Postleitzahl"]
    
    for identifier, word in text_input.items():
        if identifier in identifier_coordinates:
            if identifier in special_spacing_identifiers:
                x, y = identifier_coordinates[identifier]
                spacing = 12  # Adjust this value based on your requirement
                for i, char in enumerate(word):
                    c.drawString(x + i * spacing, y, char)
            elif isinstance(identifier_coordinates[identifier], tuple):
                c.drawString(*identifier_coordinates[identifier], f"{word}")
        else:
            print(f"Warning: Identifier {identifier} not found in mapping.")

    c.save()

    new_pdf = PyPDF2.PdfWriter()
    new_pdf.add_page(existing_page)

    new_pdf.pages[0].merge_page(PyPDF2.PdfReader(packet).pages[0])
    output.add_page(new_pdf.pages[0])

    with open(output_pdf, "wb") as f:
        output.write(f)

# Example usage:
input_pdf_form = "android_app/printout/ormblatt_1.pdf"
output_pdf_form = "android_app/printout/output_form.pdf"
input_data = {
    "Ausbildung_Klasse": "Wait",
    "Ausbildung_Antrag_gestellt_ja": "Wait",
    "Ausbildung_Antrag_gestellt_nein": "Wait",
    "Ausbildung_Amt": "Wait",
    "Ausbildung_Foerderungsnummer": "Wait",
    "Ausbilung_Abschluss": "Wait",
    "Ausbildung_Vollzeit": "Wait",
    "Ausbildung_Teilzeit": "Wait",
    "Ausbildung_Staette": "Wait",
    "Person_Geburtsort": "Wait",
    "Person_maennlich": "Wait",
    "Person_Geburtsdatum": "Wait",
    "Person_weiblich": "Wait",
    "Person_divers": "Wait",
    "Person_Name": "Wait",
    "Person_Familienstand": "Wait",
    "Person_Vorname": "Wait",
    "Person_Geburtsname": "Wait",
    "Person_Familienstand_seit": "Wait",
    "Person_Stattsangehörigkeit_eigene": "Wait",
    "Person_Stattsangehörigkeit_Ehegatte": "Wait",
    "Person_Kinder": "Wait",
    "Wohnsitz_Strasse": "Wait",
    "Wohnsitz_Land": "Wait",
    "Wohnsitz_Postleitzahl": "Wait",
    "Wohnsitz_Hausnummer": "Wait",
    "Wohnsitz_Adresszusatz": "Wait",
    "Wohnsitz_Ort": "Wait",
    "Wohnsitz_waehrend_Ausbildung_Strasse": "Wait",
    "Wohnsitz_waehrend_Ausbildung_Hausnummer": "Wait",
    "Wohnsitz_waehrend_Ausbildung_Land": "Wait",
    "Wohnsitz_waehrend_Ausbildung_Ort": "Wait",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein": "Wait",
    "Wohnsitz_waehrend_Ausbildung_Adresszusatz": "Wait",
    "Wohnsitz_waehrend_Ausbildung_Postleitzahl": "Wait",
    "Wohnsitz_waehrend_Ausbildung_elternmiete": "Wait",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja": "Wait",
    "Wohnsitz_waehrend_Ausbildung_elternmiete_nein": "Wait",
}

fill_pdf_form(input_pdf_form, output_pdf_form, input_data)
