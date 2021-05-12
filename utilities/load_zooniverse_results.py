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
        idx = new_row.index[0]
        new_row.loc[idx, 'human_transcription'], count, total = vote(subset, 'human_transcription')
        new_row.loc[idx, 'confidence'] = count/total
        if total == 3 and count == 0:
            new_row.loc[idx, 'status'] = 'Expert Required'
        elif total == 3:
            new_row.loc[idx, 'status'] = 'Complete'
        new_row.loc[idx, 'unclear'], _, _ = vote(subset, 'unclear')
        new_row.loc[idx, 'handwritten'], _, _ = vote(subset, 'handwritten')
        # discard any results where the majority voted for unclear & blank
        if new_row.loc[idx, 'status'] == 'Complete':
            if new_row.loc[idx, 'human_transcription'] == '':
                new_row.loc[idx, 'status'] = 'Discard'
            elif len(new_row.loc[idx, 'human_transcription']) > 1:
                new_row.loc[idx, 'status'] = 'Expert Required'
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
    df.loc[df['id'] == 'C0603620F-b1p0w0s1', ('human_transcription', 'status')] = ('h', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601389F-b1p0w0s9', ('human_transcription', 'status')] = ('s', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602760F-b1p0w8s4', ('human_transcription', 'status')] = ('.', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602760F-b1p0w12s0', ('human_transcription', 'status')] = ('B', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604052F-b1p0w15s0', ('human_transcription', 'status')] = ('s', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604908F-b1p1w18s1', ('human_transcription', 'status')] = ('i', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605417F-b0p0w10s1', ('human_transcription', 'status')] = ('n', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605417F-b0p0w11s4', ('human_transcription', 'status')] = ('k', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605417F-b0p1w0s1', ('human_transcription', 'status')] = ('o', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605417F-b0p1w4s0', ('human_transcription', 'status')] = ('2', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601164F-b1p0w0s3', ('human_transcription', 'status')] = ('l', 'Expert Reviewed')
    df.loc[df['id'] == 'C0045392F-b1p0w0s3', ('human_transcription', 'status')] = ('o', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604052F-b1p0w0s5', ('human_transcription', 'status')] = ('y', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602626F-b1p0w0s6', ('human_transcription', 'status')] = ('u', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602760F-b1p0w3s0', ('human_transcription', 'status')] = ('C', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602760F-b1p0w5s3', ('human_transcription', 'status')] = ('a', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602760F-b1p0w12s1', ('human_transcription', 'status')] = ('a', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602760F-b1p0w12s2', ('human_transcription', 'status')] = ('g', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601392F-b1p0w15s2', ('human_transcription', 'status')] = ('1', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601392F-b1p0w20s0', ('human_transcription', 'status')] = (',', 'Expert Reviewed')
    df.loc[df['id'] == 'C0603620F-b1p1w0s4', ('human_transcription', 'status')] = ('w', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605403F-b1p1w3s0', ('human_transcription', 'status')] = ('A', 'Expert Reviewed')
    df.loc[df['id'] == 'C0603620F-b1p1w5s0', ('human_transcription', 'status')] = ('s', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601389F-b1p1w7s0', ('human_transcription', 'status')] = ('P', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601389F-b1p1w14s0', ('human_transcription', 'status')] = ('1', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604908F-b1p1w24s3', ('human_transcription', 'status')] = ('t', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604948F-b1p2w0s1', ('human_transcription', 'status')] = ('o', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604948F-b1p2w8s0', ('human_transcription', 'status')] = ('s', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604948F-b1p2w15s5', ('human_transcription', 'status')] = ('l', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601164F-b1p3w1s0', ('human_transcription', 'status')] = ('3', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605403F-b1p3w5s3', ('human_transcription', 'status')] = ('.', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604948F-b2p0w0s0', ('human_transcription', 'status')] = ('S', 'Expert Reviewed')
    df.loc[df['id'] == 'C0603620F-b2p0w2s1', ('human_transcription', 'status')] = ('8', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p3w0s1', ('human_transcription', 'status')] = ('p', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604948F-b3p0w0s1', ('human_transcription', 'status')] = ('f', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605417F-b0p0w9s4', ('human_transcription', 'status')] = ('l', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605417F-b0p0w23s1', ('human_transcription', 'status')] = ('o', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604052F-b1p0w0s1', ('human_transcription', 'status')] = ('o', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604908F-b1p0w0s4', ('human_transcription', 'status')] = ('l', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601164F-b1p0w0s6', ('human_transcription', 'status')] = ('i', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601389F-b1p0w1s2', ('human_transcription', 'status')] = ('i', 'Expert Reviewed')
    df.loc[df['id'] == 'C0045392F-b1p0w1s5', ('human_transcription', 'status')] = ('i', 'Expert Reviewed')
    df.loc[df['id'] == 'C0045392F-b1p0w1s8', ('human_transcription', 'status')] = ('e', 'Expert Reviewed')
    df.loc[df['id'] == 'C0045392F-b1p0w1s9', ('human_transcription', 'status')] = ('r', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604052F-b1p0w1s10', ('human_transcription', 'status')] = ('i', 'Expert Reviewed')
    df.loc[df['id'] == 'C0604052F-b1p0w2s0', ('human_transcription', 'status')] = ('M', 'Expert Reviewed')
    df.loc[df['id'] == 'C0601392F-b1p0w17s0', ('human_transcription', 'status')] = ('s', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602626F-b1p1w15s0', ('human_transcription', 'status')] = (',', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605403F-b1p3w1s0', ('human_transcription', 'status')] = ('R', 'Expert Reviewed')
    df.loc[df['id'] == 'C0605403F-b1p4w2s1', ('human_transcription', 'status')] = ('0', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p0w0s6', ('human_transcription', 'status')] = ('i', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p0w2s0', ('human_transcription', 'status')] = ('D', 'Expert Reviewed')
    df.loc[df['id'] == 'C0603614F-b2p0w30s1', ('human_transcription', 'status')] = ('e', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p1w4s1', ('human_transcription', 'status')] = ('o', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p1w8s3', ('human_transcription', 'status')] = ('l', 'Expert Reviewed')
    df.loc[df['id'] == 'C0602766F-b2p3w0s0', ('human_transcription', 'status')] = ('A', 'Expert Reviewed')

    df.loc[df['id'] == 'C0602626F-b1p0w0s6', ('unclear', 'status')] = (True, 'Discard')
    df.loc[df['id'] == 'C0604052F-b1p0w2s1', ('unclear', 'status')] = (True, 'Discard')

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
    df.loc[df['id'] == 'C0045392F-b6p0w3s0', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0045392F-b1p0w1s1', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604052F-b1p0w2s1', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604052F-b1p0w2s3', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0601392F-b1p0w15s4', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0601392F-b1p1w0s4', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604948F-b1p2w12s3', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0603620F-b2p0w0s1', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0603620F-b2p1w3s0', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0602766F-b2p1w5s5', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0602766F-b2p1w6s5', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604948F-b6p0w1s1', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0605417F-b0p1w0s0', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0601164F-b1p0w1s2', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0602626F-b1p0w1s9', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604052F-b1p0w1s11', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604052F-b1p0w2s2', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0602626F-b1p1w10s0', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0604948F-b1p2w12s1', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0603614F-b2p0w1s4', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0602766F-b2p1w1s0', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0602766F-b2p2w0s0', 'status'] = 'Discard - Reviewed'
    df.loc[df['id'] == 'C0045392F-b6p0w3s0', 'status'] = 'Discard - Reviewed'


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'Include 2 arguments: (1) the location of the classification results from Zooniverse, ' + \
                               'and (2) the folder of images of letters.'
    zooniverse = sys.argv[1]  # 'file_resources\\herbarium-handwriting-transcription-classifications.csv'  # sys.argv[1]
    assert os.path.isfile(zooniverse), 'Invalid 1st argument: must be a file on the local computer.'
    image_folder = sys.argv[2]  # 'file_resources\\gcv_letter_images'  # sys.argv[2]
    assert os.path.isdir(image_folder), 'Invalid 2nd argument: must be a folder on the local computer.'
    main(zooniverse, image_folder)
