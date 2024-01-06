
def parse_information_from_file(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        text = file.read()
    sections = [section.strip() for section in text.split('\n\n') if section.strip()]

    data_dict = {}

    for section in sections:
        lines = section.split('\n')

        for line in lines:
            if '-' in line:
                key, value = [item.strip() for item in line.split(':', 1)]
                data_dict[key] = value

    return data_dict

if __name__ == '__main__':
    file_path = 'example.txt'
    result = parse_information_from_file(file_path)
    print(result)