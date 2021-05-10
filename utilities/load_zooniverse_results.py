import os
import sys
import ast
import pandas as pd
from statistics import mode, StatisticsError
from dataloader import save_dataframe_as_csv


def main(zooniverse_classifications_path: str, image_folder_path: str):
    raw_zooniverse_classifications = pd.read_csv(zooniverse_classifications_path)
    zooniverse_classifications = parse_raw_zooniverse_file(raw_zooniverse_classifications)
    zooniverse_classifications = consolidate_classifications(zooniverse_classifications)
    load_letter_images(image_folder_path, zooniverse_classifications)
    expert_manual_review(zooniverse_classifications)
    save_location = save_dataframe_as_csv('file_resources', 'zooniverse_parsed', zooniverse_classifications)
    print('Saved to %s' % save_location)


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
    parsed_zooniverse_classifications['status'] = 'In Progress'
    return parsed_zooniverse_classifications


def consolidate_classifications(zooniverse_classifications: pd.DataFrame) -> pd.DataFrame:
    duplicates = zooniverse_classifications[zooniverse_classifications['seen_count'] > 1]
    ids = set(duplicates['image_location'])
    for id_name in ids:
        subset = duplicates[duplicates['image_location'] == id_name]
        new_row = subset.head(1).copy()
        new_row.loc[:, 'human_transcription'], count, total = vote(subset, 'human_transcription')
        new_row.loc[:, 'confidence'] = count/total
        if total == 3 and count == 0:
            new_row.loc[:, 'status'] = 'Expert Required'
        elif total == 3:
            new_row.loc[:, 'status'] = 'Complete'
        new_row.loc[:, 'unclear'], _, _ = vote(subset, 'unclear')
        new_row.loc[:, 'handwritten'], _, _ = vote(subset, 'handwritten')
        # discard any results where the majority voted for unclear & blank
        if (new_row.loc[:, 'status'] == 'Complete').all() and (new_row.loc[:, 'human_transcription'] == '').all():
            new_row.loc[:, 'status'] = 'Discard'
        zooniverse_classifications = zooniverse_classifications.drop(subset.index)
        zooniverse_classifications = zooniverse_classifications.append(new_row)
        # print('Group %s vote is %s with a confidence of %.0f' %
        #       (id_name, new_row['human_transcription'].values[0], (new_row['confidence'].values[0])*100) + '%.')
    return zooniverse_classifications.sort_values(by=['block', 'paragraph', 'word', 'symbol'], ascending=True)


def vote(df: pd.DataFrame, col_name: str) -> (any, int, int):
    total = df.shape[0]
    if total > 3:
        df = df[-3:]
        total = 3
    try:
        voted = mode(list(df.loc[:, col_name]))  # todo: Note if upgrade to Python 3.8, can use multimode instead
        voted_count = df[df[col_name] == voted].shape[0]
    except StatisticsError as se:
        # If there's a tie
        # Option 1: It's a 1-1 vote on 2 images. Arbitrarily pick the first option.
        if total == 2:
            voted = list(df.loc[:, col_name])[0]
            voted_count = 1
        # Option 2: It's a 1-1-1 tie on 3 images. Flag for expert review.
        else:
            voted = str(list(df.loc[:, col_name]))
            voted_count = 0

    return voted, voted_count, total


def load_letter_images(image_folder_path: str, zooniverse_classifications: pd.DataFrame) -> None:
    for idx, row in zooniverse_classifications.iterrows():
        image_name = row['image_location']
        if not os.path.isfile(os.path.join(image_folder_path, image_name)):
            print('Warning: %s doesn\'t exist in this location.' % image_name)
        zooniverse_classifications.at[idx, 'image_location'] = os.path.join(image_folder_path, image_name)


def expert_manual_review(df: pd.DataFrame) -> None:
    df.loc[df['id'] == 'C0603620F-b1p0w0s8', ('human_transcription', 'status')] = ('r', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602626F-b1p0w0s6', ('unclear', 'status')] = (True, 'Discard')
    df.loc[df['id'] == 'C0601389F-b1p0w1s8', ('human_transcription', 'confidence', 'status')] = \
        ('l', 1, 'Expert Reviewed')
    df.loc[df['id'] == 'C0604908F-b1p0w1s8', ('human_transcription', 'confidence', 'status')] = \
        ('i', 1, 'Expert Reviewed')
    df.loc[df['id'] == 'C0604948F-b1p0w3s18', ('human_transcription', 'confidence', 'status')] = \
        ('k', 1, 'Expert Reviewed')
    df.loc[df['id'] == 'C0604908F-b1p1w1s0', ('human_transcription', 'confidence', 'status')] = \
        ('f', 1, 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p1w1s1', ('human_transcription', 'confidence', 'status')] = \
        ('o', 1, 'Expert Reviewed')
    df.loc[df['id'] == 'C0601389F-b1p0w1s10', ('unclear', 'status')] = \
        (False, 'Expert Reviewed')
    df.loc[df['id'] == 'C0601389F-b1p2w1s0', ('unclear', 'status')] = \
        (False, 'Expert Reviewed')
    df.loc[df['id'] == 'C0045392F-b6p0w3s0', 'status'] = 'Discard'


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Include 2 arguments: (1) the location of the classification results from Zooniverse, ' + \
                               'and (2) the folder of images of letters.'
    zooniverse = sys.argv[1]  # 'file_resources\\herbarium-handwriting-transcription-classifications.csv'  # sys.argv[1]
    assert os.path.isfile(zooniverse), 'Invalid 1st argument: must be a file on the local computer.'
    image_folder = sys.argv[2]  # 'file_resources\\gcv_letter_images'  # sys.argv[2]
    assert os.path.isdir(image_folder), 'Invalid 2nd argument: must be a folder on the local computer.'
    main(zooniverse, image_folder)
