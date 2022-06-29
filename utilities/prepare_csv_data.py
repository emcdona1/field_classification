import os
import sys
import json
import pandas as pd
from pathlib import Path
from timer import Timer
from dataloader import save_dataframe_as_csv


WORKFLOW_NAME = 'Determining the Reproductive Structure of a Liverwort'
MINIMUM_WORKFLOW_VERSION = 51.690


def main(given_file: Path):
    timer = Timer('Filter CSV Data')
    assert os.path.isfile(given_file), f'Invalid 1st argument: {given_file} is not a file.'
    given_data = pd.read_csv(given_file, dtype={"user_id": pd.Int64Dtype()})
    given_data = given_data.drop(columns=[
        'user_name', 'user_ip',
        'created_at', 'gold_standard', 'expert',
        'metadata'
    ])
    given_data = _filter_workflow_versions(given_data)
    # given_data = given_data.reset_index(drop=True)
    given_data = _expand_dict_columns(given_data)
    given_data = given_data.rename(columns={'annotations': 'user_answer', 'subject_data': 'image_file'})
    results_dir = Path('utilities/saved_csvs')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_dataframe_as_csv(results_dir, 'altered_csv', given_data)
    timer.stop()
    timer.print_results()


def _filter_workflow_versions(raw_data: pd.DataFrame) -> pd.DataFrame:
    filtered_data = raw_data.query(
        f'workflow_name == "{WORKFLOW_NAME}" and workflow_version >= {MINIMUM_WORKFLOW_VERSION}').copy()
    return filtered_data

def _expand_dict_columns(given_df: pd.DataFrame) -> pd.DataFrame:
    def clean_ann(ann: str, ask_rect: bool) -> (str, bool):
        ann_out = ""
        rect_out = ask_rect
        curr = json.loads(ann)
        for val in curr:
            if val['task'] == 'T0':
                ann_out = val['value']
            elif val['task'] != 'T1':
                rect_out = True
        return ann_out, rect_out

    def clean_sub(sub: str) -> (str):
        curr = json.loads(sub)
        [(s1, s2)] = curr.items()
        return s2['Filename']

    column_name = 'asked_for_rectangle'
    if column_name not in given_df.columns:
        given_df[column_name] = False

    given_df['annotations'] = given_df.apply(lambda x: clean_ann(x.annotations, x.asked_for_rectangle), axis=1)
    given_df[['annotations', 'asked_for_rectangle']] = pd.DataFrame(given_df.annotations.tolist(), index=given_df.index)
    given_df['subject_data'] = given_df.apply(lambda x: clean_sub(x.subject_data), axis=1)
    return given_df

if __name__ == '__main__':
    given_file = Path(sys.argv[1])
    main(given_file)