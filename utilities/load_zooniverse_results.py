import os
import sys
import ast
import pandas as pd
from statistics import mode, StatisticsError
from utilities.dataloader import save_dataframe_as_csv


def main(zooniverse_classifications_path: str, image_folder_path: str):
    raw_zooniverse_classifications = pd.read_csv(zooniverse_classifications_path)
    zooniverse_classifications = parse_raw_zooniverse_file(raw_zooniverse_classifications)
    zooniverse_classifications = consolidate_classifications(zooniverse_classifications)
    load_letter_images(image_folder_path, zooniverse_classifications)
    save_dataframe_as_csv('file_resources', 'zooniverse_parsed', zooniverse_classifications)


def parse_raw_zooniverse_file(raw_zooniverse_classifications: pd.DataFrame) -> pd.DataFrame:
    filtered_raw_zooniverse = raw_zooniverse_classifications.query(
        'workflow_id == 17842 and workflow_version >= 25.103').copy()
    filtered_raw_zooniverse.loc[:, 'annotations'] = filtered_raw_zooniverse['annotations'].apply(ast.literal_eval)

    def clean_subject_data(sd: str):
        sd = sd.replace('null', 'None')
        sd = ast.literal_eval(sd)
        subject_list = [*sd.values()]
        return subject_list[0]

    filtered_raw_zooniverse.loc[:, 'subject_data'] = filtered_raw_zooniverse['subject_data']\
        .apply(clean_subject_data)

    parsed_zooniverse_classifications = pd.DataFrame()
    parsed_zooniverse_classifications['id'] = filtered_raw_zooniverse['subject_data'].apply(
        lambda annotation: annotation['image_of_boxed_letter'].replace('symbox-', '').replace('.jpg', '').replace(
            'label-', ''))

    def parse_subject(s):
        barcode = s['barcode'].split('-')[0]
        image_name = s['image_of_boxed_letter'].replace('symbox', 'symbol')
        col_names = ['barcode', 'block', 'paragraph', 'word', 'symbol', 'gcv_identification', 'image_location']
        result = pd.Series([barcode, s['block_no'], s['paragraph_no'], s['word_no'], s['symbol_no'],
                          s['#GCV_identification'], image_name], index=col_names)
        return result

    location = filtered_raw_zooniverse['subject_data'].apply(parse_subject)
    parsed_zooniverse_classifications = pd.concat([parsed_zooniverse_classifications, location], axis=1)
    parsed_zooniverse_classifications['handwritten'] = filtered_raw_zooniverse['annotations'].apply(
        lambda annotation: annotation[0]['value'] == 'handwritten')
    parsed_zooniverse_classifications['human_transcription'] = filtered_raw_zooniverse['annotations'].apply(
        lambda annotation: annotation[1]['value'])
    parsed_zooniverse_classifications['unclear'] = parsed_zooniverse_classifications['human_transcription'].apply(
        lambda transcription: '[unclear][/unclear]' in transcription)
    parsed_zooniverse_classifications['human_transcription'] = parsed_zooniverse_classifications['human_transcription']\
        .apply(lambda transcription: transcription.replace('[unclear][/unclear]', ''))

    parsed_zooniverse_classifications['seen_count'] = parsed_zooniverse_classifications.groupby('id')[
        'block'].transform(len)
    parsed_zooniverse_classifications['confidence'] = 1.0
    parsed_zooniverse_classifications['block'] = pd.to_numeric(parsed_zooniverse_classifications['block'])
    parsed_zooniverse_classifications['paragraph'] = pd.to_numeric(parsed_zooniverse_classifications['paragraph'])
    parsed_zooniverse_classifications['word'] = pd.to_numeric(parsed_zooniverse_classifications['word'])
    parsed_zooniverse_classifications['symbol'] = pd.to_numeric(parsed_zooniverse_classifications['symbol'])
    return parsed_zooniverse_classifications


def vote(df: pd.DataFrame, col_name: str) -> (any, float):
    try:
        voted = mode(list(df.loc[:, col_name]))  # todo: Note if upgrade to Python 3.8, can use multimode instead
    except StatisticsError as se:
        # If there's a tie, it has to be 1-1 (max 3 votes/image).  So, arbitrarily pick the first option.
        voted = list(df.loc[:, col_name])[0]
    voted_count = df[df[col_name] == voted].shape[0]
    total = df.shape[0]
    return voted, voted_count/total


def consolidate_classifications(zooniverse_classifications: pd.DataFrame) -> pd.DataFrame:
    duplicates = zooniverse_classifications[zooniverse_classifications['seen_count'] > 1]
    ids = set(duplicates['image_location'])
    for id_name in ids:
        subset = duplicates[duplicates['image_location'] == id_name]
        new_row = subset.head(1).copy()
        new_row.loc[:, 'human_transcription'], new_row.loc[:, 'confidence'] = vote(subset, 'human_transcription')
        new_row.loc[:, 'unclear'], _ = vote(subset, 'unclear')
        new_row.loc[:, 'handwritten'], _ = (subset, 'handwritten')

        zooniverse_classifications = zooniverse_classifications.drop(subset.index)
        zooniverse_classifications = zooniverse_classifications.append(new_row)
        print('Group %s vote is %s with a confidence of %.0f' %
              (id_name, new_row['human_transcription'].values[0], (new_row['confidence'].values[0])*100) + '%.')
    return zooniverse_classifications.sort_values(by=['block', 'paragraph', 'word', 'symbol'], ascending=True)


def load_letter_images(image_folder_path: str, zooniverse_classifications: pd.DataFrame) -> None:
    for idx, row in zooniverse_classifications.iterrows():
        image_name = row['image_location']
        if not os.path.isfile(os.path.join(image_folder_path, image_name)):
            print('Warning: %s doesn\'t exist in this location.' % image_name)
        zooniverse_classifications.at[idx, 'image_location'] = os.path.join(image_folder_path, image_name)


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Include 2 arguments: (1) the location of the classification results from Zooniverse, ' + \
                               'and (2) the folder of images of letters.'
    zooniverse = 'file_resources\\herbarium-handwriting-transcription-classifications.csv'  # sys.argv[1]
    assert os.path.isfile(zooniverse), 'Invalid 1st argument: must be a file on the local computer.'
    image_folder = 'file_resources\\gcv_letter_images'  # sys.argv[2]
    assert os.path.isdir(image_folder), 'Invalid 2nd argument: must be a folder on the local computer.'
    main(zooniverse, image_folder)
