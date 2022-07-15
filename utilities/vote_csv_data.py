import pandas as pd
from statistics import multimode
from timer import Timer
import scipy


def vote(df: pd.DataFrame, col_name: str) -> (list, int, int, int):
    total = df.shape[0]
    not_sure = df[df[col_name] == 'NotSure'].shape[0]

    voted = multimode(list(df[df[col_name].isin(['Male', 'Female', 'Both', 'Sterile'])].loc[:, col_name]))
    # mode = scipy.stats.mode(list(df.loc[:, col_name]))
    # df[df[col_name].isin(['Male', 'Female', 'Both', 'Sterile'])]

    if type(voted) is not list:
        voted = [voted]
    voted_count = df[df[col_name] == voted[0]].shape[0]
    # print(voted)
    # print(voted_count)
    # print(total)
    # print(not_sure)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    return voted, voted_count, total, not_sure


def _vote_on_sex(subset, consolidated_row):
    sex_vote, correct_votes, total_votes, not_sure_votes = vote(subset, 'sex_determined_by_user')
    consolidated_row.at['voted_sex'] = sex_vote[0]
    consolidated_row.at['confidence'] = float(correct_votes/total_votes)
    consolidated_row.at['percent_not_sure'] = float(not_sure_votes/total_votes)
    consolidated_row.at['num_of_votes'] = total_votes


def _filter_duplicates_after_voting(voted: list) -> list:
    counts = {a: voted.count(a) for a in voted}
    max_count = max(counts.values())
    voted = [a for a in voted if voted.count(a) == max_count]
    return list(set(voted))


def vote_on_results(voted_df: pd.DataFrame) -> pd.DataFrame:
    c = Timer('voting')
    all_unique_ids = set(voted_df['image_file'])
    for image_id in all_unique_ids:
        subset = voted_df[voted_df['image_file'] == image_id]
        consolidated_row = subset.loc[subset.index[0], :]
        # print('===')
        # print(consolidated_row['subject_ids'])
        # print('===')
        _vote_on_sex(subset, consolidated_row)
        voted_df = voted_df.drop(index=subset.index)
        voted_df = voted_df.append(consolidated_row)
        # voted_df['voted_sex'][subset.index] = consolidated_row['voted_sex']
    c.stop()
    return voted_df
