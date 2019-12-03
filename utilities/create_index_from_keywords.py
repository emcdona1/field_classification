import os
import csv


def load_file_and_create_dictionary(filename):
    index_dictionary = {}
    with open(filename, 'r') as csvreader:
        reader = csv.reader(csvreader)
        next(reader, None) #skip header
        for row in reader:
            keyword = row[0]
            reference = clean_name(row[1])
            if keyword in index_dictionary:
                current_references = index_dictionary[keyword]
                index_dictionary[keyword] = current_references + ', ' + reference
            else:
                index_dictionary[keyword] = reference
    return index_dictionary


def clean_name(string):
    arr = string.split('_')
    return arr[0]


def write_index_to_tsv(dictionary):
    with open('bryophyte_index.tsv', 'w') as file:
        for key in dictionary:
            file.write(key)
            file.write('\t')
            file.write(dictionary[key])
            file.write('\n')


if __name__ == '__main__':
    filename = input('Type in file name: ')
    index_dictionary = load_file_and_create_dictionary(filename)
    write_index_to_tsv(index_dictionary)