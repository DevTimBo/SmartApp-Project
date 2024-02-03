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
    c.setFont("Helvetica", 11) 

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
        "Wohnsitz_waehrend_Ausbildung_Hausnummer": (231.5, 77),
        "Wohnsitz_waehrend_Ausbildung_Adresszusatz": (290, 77),
        "Wohnsitz_waehrend_Ausbildung_Land": (58, 47),
        "Wohnsitz_waehrend_Ausbildung_Postleitzahl": (93, 47),
        "Wohnsitz_waehrend_Ausbildung_Ort": (160, 47),
    }
    
    special_spacing_identifiers = ["Ausbildung_Foerderungsnummer", "Person_Geburtsdatum", "Person_Familienstand_seit",
                                   "Wohnsitz_Hausnummer", "Wohnsitz_Land", "Wohnsitz_Postleitzahl",
                                   "Wohnsitz_waehrend_Ausbildung_Hausnummer", "Wohnsitz_waehrend_Ausbildung_Land", "Wohnsitz_waehrend_Ausbildung_Postleitzahl"]
    
    special_tof_identifiers = ["Ausbildung_Antrag_gestellt_ja", "Ausbildung_Antrag_gestellt_nein", "Ausbildung_Vollzeit", "Ausbildung_Teilzeit",
                               "Person_Kinder", "Wohnsitz_waehrend_Ausbildung_elternmiete", "Wohnsitz_waehrend_Ausbildung_elternmiete_nein",
                               "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja", "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein"
                               ]
    
    for identifier, word in text_input.items():
        if identifier in identifier_coordinates:
            if identifier in special_spacing_identifiers:
                x, y = identifier_coordinates[identifier]
                spacing = 11.25  # Adjust this value based on your requirement
                for i, char in enumerate(word):
                    c.drawString(x + i * spacing, y, char)
            elif identifier in special_tof_identifiers:
                if word == "Ja":
                    c.drawString(*identifier_coordinates[identifier], "X")
                else:
                    c.drawString(*identifier_coordinates[identifier], " ")
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
    "Ausbildung_Klasse": "Theologie",
    "Ausbildung_Antrag_gestellt_ja": "",
    "Ausbildung_Antrag_gestellt_nein": "Ja",
    "Ausbildung_Amt": "Bremen Mitte",
    "Ausbildung_Foerderungsnummer": "081234995821",
    "Ausbilung_Abschluss": "Master",
    "Ausbildung_Vollzeit": "Ja",
    "Ausbildung_Teilzeit": "",
    "Ausbildung_Staette": "Universität Bremen",
    "Person_Geburtsort": "Bremen",
    "Person_maennlich": "X",
    "Person_Geburtsdatum": "09201999",
    "Person_weiblich": "",
    "Person_divers": "",
    "Person_Name": "Thompson",
    "Person_Familienstand": "ledig",
    "Person_Vorname": "Josephine",
    "Person_Geburtsname": "Christina",
    "Person_Familienstand_seit": "19091999",
    "Person_Stattsangehörigkeit_eigene": "Deutsch",
    "Person_Stattsangehörigkeit_Ehegatte": "Russisch",
    "Person_Kinder": "Ja",
    "Wohnsitz_Strasse": "Butjadingerstrasse 1",
    "Wohnsitz_Land": "GER",
    "Wohnsitz_Postleitzahl": "28195",
    "Wohnsitz_Hausnummer": "15",
    "Wohnsitz_Adresszusatz": "im Hinterhof",
    "Wohnsitz_Ort": "Bremen",
    "Wohnsitz_waehrend_Ausbildung_Strasse": "Schwachhauser Heerstrasse 1",
    "Wohnsitz_waehrend_Ausbildung_Hausnummer": "19",
    "Wohnsitz_waehrend_Ausbildung_Land": "GER",
    "Wohnsitz_waehrend_Ausbildung_Ort": "Bremen",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein": "Ja",
    "Wohnsitz_waehrend_Ausbildung_Adresszusatz": "im vorderen Haus",
    "Wohnsitz_waehrend_Ausbildung_Postleitzahl": "28209",
    "Wohnsitz_waehrend_Ausbildung_elternmiete": "Ja",
    "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja": "",
    "Wohnsitz_waehrend_Ausbildung_elternmiete_nein": "Wait",
}

fill_pdf_form(input_pdf_form, output_pdf_form, input_data)
