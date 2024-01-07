
def parse_information_from_file(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        text = file.read()
    sections = [section.strip() for section in text.split('\n\n') if section.strip()]

    full_dict = {}

    for i, section in enumerate(sections):
        lines = section.split('\n')
        if i == 0:
            title = lines[0].split(":")
            full_dict['Title'] = title
        elif i == 1:
            full_dict['Ausbildung'] = section_scanner(section)
        elif i == 2:
            full_dict['Person'] = section_scanner(section)
        elif i == 3:
            full_dict['Wohnsitz_w_Ausbildung'] = section_scanner(section)
    return full_dict

def section_scanner(section):
    temp_dict = {}
    for line in section.split('\n'):
        line = line.strip()
        if line.startswith('-'):
            parts = line[1:].split(':', 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            temp_dict[key] = value
            #print(f"{key} : {temp_dict[key]}")

    return temp_dict


sub_class_ids = [
"Ausbildung_Staette" ,
"Ausbildung_Klasse" ,
"Ausbilung_Abschluss" ,
"Ausbildung_Vollzeit" ,
"Ausbildung_Teilzeit" ,
"Ausbildung_Antrag_gestellt_ja" ,
"Ausbildung_Antrag_gestellt_nein" ,
"Ausbildung_Amt" ,
"Ausbildung_Foerderungsnummer" ,

"Person_Name",
"Person_Vorname" ,
"Person_Geburtsname" ,
"Person_Geburtsort" ,
"Person_maennlich" ,
"Person_weiblich",
"Person_divers",
"Person_Geburtsdatum" ,
"Person_Familienstand" ,
"Person_Familienstand_seit",
"Person_Stattsangehörigkeit_eigene" ,
"Person_Stattsangehörigkeit_Ehegatte" ,
"Person_Kinder",

"Wohnsitz_Strasse",
"Wohnsitz_Hausnummer",
"Wohnsitz_Adresszusatz",
"Wohnsitz_Land",
"Wohnsitz_Postleitzahl",
"Wohnsitz_Ort",

#
# "Ausbildung" ,
# "Wohnsitz_waehrend_Ausbildung_Strasse",
# "Wohnsitz_waehrend_Ausbildung_Hausnummer",
# "Wohnsitz_waehrend_Ausbildung_Land",
# "Wohnsitz_waehrend_Ausbildung_Ort",
# "Wohnsitz_waehrend_Ausbildung_elternwohnung_nein",
# "Wohnsitz_waehrend_Ausbildung_Adresszusatz",
# "Wohnsitz_waehrend_Ausbildung_Postleitzahl",
# "Wohnsitz_waehrend_Ausbildung_elternmiete",
# "Wohnsitz_waehrend_Ausbildung_elternwohnung_ja",
# "Wohnsitz_waehrend_Ausbildung_elternmiete_nein"
]



if __name__ == '__main__':
    file_path = 'example.txt'
    result = parse_information_from_file(file_path)
    print(result['Person']["Name"])