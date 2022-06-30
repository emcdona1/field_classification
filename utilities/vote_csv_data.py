import pandas as pd
from statistics import multimode
from timer import Timer


def vote(df: pd.DataFrame, col_name: str) -> (list, int, int):
    total = df.shape[0]
    voted = multimode(list(df.loc[:, col_name]))
    if type(voted) is not list:
        voted = [voted]
    voted_count = df[df[col_name] == voted[0]].shape[0]
    return voted, voted_count, total


def _vote_on_sex(subset, consolidated_row):
    type_vote, _, _ = vote(subset, 'sex_determined_by_user')
    consolidated_row.at['voted_sex'] = type_vote[0]


def _filter_duplicates_after_voting(voted: list) -> list:
    counts = {a: voted.count(a) for a in voted}
    max_count = max(counts.values())
    voted = [a for a in voted if voted.count(a) == max_count]
    return list(set(voted))


def vote_on_results(voted_df: pd.DataFrame) -> pd.DataFrame:
    c = Timer('voting')
    all_unique_ids = set(voted_df['subject_ids'])
    for image_id in all_unique_ids:
        subset = voted_df[voted_df['subject_ids'] == image_id]
        consolidated_row = subset.loc[subset.index[0], :]
        _vote_on_sex(subset, consolidated_row)
        voted_df['voted_sex'][subset.index] = consolidated_row['voted_sex']
    c.stop()
    return voted_df
