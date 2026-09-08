"""Utilities for joining dataframes via externally supplied match data.

This module contains functions that join pandas dataframes using value
correspondences read from an external source, rather than requiring the
join columns to match exactly or recomputing the correspondence each time.
"""

import json

import pandas as pd


def json_join(
    json_path: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_key: str = "left_value",
    right_key: str = "right_value",
    how: str = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """
    Joins two dataframes using value-to-value matches read from a JSON file,
    instead of requiring left_on and right_on to match exactly. Useful for
    replaying a correspondence produced elsewhere (manual curation, a fuzzy
    matcher, an LLM, ...) without recomputing it.

    Inputs:
        json_path -> str: Path to a JSON file holding a list of records,
                          each describing one matched value pair
        left -> pd.DataFrame: Left dataframe
        right -> pd.DataFrame: Right dataframe
        left_on -> str: Column in left whose values are matched against
                        left_key in the JSON records
        right_on -> str: Column in right whose values are matched against
                         right_key in the JSON records
        left_key -> str: Key in each JSON record holding the left-side value
        right_key -> str: Key in each JSON record holding the right-side
                          value
        how -> str: Merge strategy passed through to pandas ('inner',
                    'left', 'right' or 'outer')
        suffixes -> tuple[str,str]: Suffixes appended to overlapping column
                                    names from left and right, respectively

    Outputs:
        joined -> pd.DataFrame: left and right rows joined via the matches
                                found in the JSON file
    """
    with open(json_path, 'r', encoding='utf-8') as json_file:
        records = json.load(json_file)

    pairs = pd.DataFrame(
        [(record[left_key], record[right_key]) for record in records],
        columns=[left_on, right_on],
    )

    return left.merge(pairs, on=left_on, how=how).merge(
        right, on=right_on, how=how, suffixes=suffixes
    )
