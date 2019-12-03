import docx
import os
import re

def get_content_from_docx(filename):
    doc = docx.Document(filename)
    full_text = ''
    for para in doc.paragraphs:
        full_text += para.text + '\n'
    return full_text

def find_keywords(full_text, search_text):
    search_text = 'key[\s]{0,1}word[s]{0,1}:[\s]*'
    keywords = re.split(search_text, full_text.lower())[1]
    keywords = re.split('[\n]+', keywords)[0]
    keywords = re.split('[,，]{1}[\s]*', keywords)
    return keywords

def write_list(list, filename):
    with open(filename, 'w') as f:
        for item in keyword_list:
            f.write(item[0])
            f.write(',')
            f.write(item[1])
            f.write('\n')

def parse_docx(keyword_list):
    try:
        file_path = os.path.join(folder, doc)
        full_text = get_content_from_docx(file_path)
        full_text = full_text.replace('.','').replace(';', '').replace('�','')
        keywords = find_keywords(full_text, 'keywords: ')
        assert (len(keywords) > 0), ''
        for k in keywords:
            keyword_list.append([doc, k])
    except ValueError as ve:
        incompatible_files.append(doc)
        keyword_list.append([doc, 'NOT A DOCX FILE'])
    except docx.opc.exceptions.PackageNotFoundError as pnfe:
        incompatible_files.append(doc)
        keyword_list.append([doc, 'NOT A DOCX FILE'])
    except AssertionError as ae:
        incompatible_files.append(doc)
        keyword_list.append([doc, 'NO KEYWORDS FOUND'])
    except IndexError as ie:
        incompatible_files.append(doc)
        keyword_list.append([doc, '\'KEYWORDS\' NOT FOUND'])
    

if __name__ == '__main__':
    folder = input('Name of folder containing docx files: ')
    incompatible_files = []
    keyword_list = []
    for doc in os.listdir(folder):
        parse_docx(keyword_list)
    write_list(keyword_list, 'keywords.csv')
    
    print('Files with incompatible formatting: ')
    print(incompatible_files)