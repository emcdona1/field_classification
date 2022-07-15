import os
import sys
import json
import pandas as pd
from pathlib import Path
from timer import Timer
from dataloader import save_dataframe_as_csv
from vote_csv_data import vote_on_results

WORKFLOW_NAME = 'Determining the Reproductive Structure of a Liverwort'
MINIMUM_WORKFLOW_VERSION = 80.124


def main(given_file: Path):
    timer = Timer('Filter CSV Data')
    assert os.path.isfile(given_file), f'Invalid 1st argument: {given_file} is not a file.'
    (root, ext) = os.path.splitext(given_file.name)

    given_data = pd.read_csv(given_file, dtype={"user_id": pd.Int64Dtype()})
    given_data = given_data.drop(columns=[
        'user_name', 'user_ip', 'created_at', 'gold_standard', 'expert', 'metadata'
    ])
    given_data['ann_temp'] = given_data.loc[:, 'annotations']
    given_data = _filter_workflow_versions(given_data)

    given_data = _expand_dict_columns(given_data)

    # voted_data = given_data.sort_values(by=['subject_ids'])
    voted_data = given_data.rename(columns={
        'annotations': 'sex_determined_by_user',
        'subject_data': 'image_file'
    })
    voted_data = voted_data.drop(columns={'classification_id', 'user_id', 'ann_temp', 'subject_ids', 'asked_for_rectangle', 'T3_male', 'T4_mf', 'T5_female'})
    voted_data = voted_data[voted_data['sex_determined_by_user'].isin(['Male', 'Female', 'Both', 'Sterile', 'NotSure'])]
    # voted_data['voted_sex', 'confidence', 'percent_not_sure', 'num_of_votes'] = ''

    voted_data = vote_on_results(voted_data)

    voted_data = voted_data.drop(columns={'sex_determined_by_user'})
    voted_data = voted_data.sort_values(by=['image_file'])

    given_data = given_data.rename(columns={
        'annotations': 'Please identify if the image of the microplant shown best corresponds to a female, male, sterile, or both a female and a male structure.',
        'T3_male': 'Please draw a rectangle around all male  reproductive identifiers you see in the image ',
        'T4_mf': 'Please draw a rectangle around all male and female reproductive identifiers you see in the image',
        'T5_female': 'Please draw a rectangle around all female reproductive identifiers and determine where they are located in the image ',
        'subject_data': 'image_file',
        'ann_temp': 'annotations'
    })

    results_dir = Path('utilities/saved_csvs')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_dataframe_as_csv(results_dir, f'{root}-cleaned', given_data)
    save_dataframe_as_csv(results_dir, f'{root}-voted', voted_data)

    timer.stop()
    timer.print_results()


def _filter_workflow_versions(raw_data: pd.DataFrame) -> pd.DataFrame:
    filtered_data = raw_data.query(
        f'workflow_name == "{WORKFLOW_NAME}" and workflow_version >= {MINIMUM_WORKFLOW_VERSION}').copy()
    return filtered_data


def _expand_dict_columns(given_df: pd.DataFrame) -> pd.DataFrame:
    def clean_annotation(annotation: str, ask_rect: bool) -> (str, bool):
        ann_out = ''
        rect_out = ask_rect
        sex_out = {'T3': '', 'T4': '', 'T5': ''}  # T3 = male, T4 = male + female, T5 = female
        curr = json.loads(annotation)
        for val in curr:
            if val['task'] == 'T0':
                ann_out = val['value']
                if type(ann_out) == str:
                    ann_out = ann_out.replace("Both Female and Male", "Both")
                    ann_out = ann_out.replace(" ", "")
            elif val['task'] != 'T1':
                rect_out = True
                sex_out[val['task']] = val['value']  # T3 = male, T4 = male + female, T5 = female
        return ann_out, rect_out, sex_out['T3'], sex_out['T4'], sex_out['T5']

    def clean_subject_data(sub: str) -> str:
        curr = json.loads(sub)
        [(s1, s2)] = curr.items()
        return s2['Filename']

    def make_column(column_name: str, fill_with):
        if column_name not in given_df.columns:
            given_df[column_name] = fill_with

    make_column('asked_for_rectangle', False)
    make_column('T3_male', '')
    make_column('T4_mf', '')
    make_column('T5_female', '')

    given_df['annotations'] = given_df.apply(lambda x: clean_annotation(x.annotations, x.asked_for_rectangle), axis=1)
    given_df[['annotations', 'asked_for_rectangle', 'T3_male', 'T4_mf', 'T5_female']] = pd.DataFrame(
        given_df.annotations.tolist(), index=given_df.index)
    given_df['subject_data'] = given_df.apply(lambda x: clean_subject_data(x.subject_data), axis=1)
    return given_df


if __name__ == '__main__':
    g_file = Path(sys.argv[1])
    main(g_file)
