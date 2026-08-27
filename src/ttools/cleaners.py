"""Utilities for database cleaning.

This module contains functions that aid in the cleaning of pandas 
dataframes and on the handling and correcting of data types
"""

import string 
import warnings
import unicodedata

import pandas as pd
import numpy as np


from typing import Literal

def _find_not_numeric(column:pd.Series) -> np.array:
    """
    Finds fully non numeric values in an alleged numeric column.
    Inputs:
        columns -> pd.Series: The pandas series where the non numeric values are to be found
    
    Outputs:
        non-numeric -> np.array: A numpy array containing all fully non numeric values to be
                                 nulled in he subsequent processing

    """
    values = column.unique()
    non_numeric = [value for value in values if not(any(char.isdigit() for char in value))]
    return non_numeric

def _turn_numeric(col:pd.Series,
                  replace_comma_with: Literal['','.']) -> pd.Series:
    """
    Recieves a non-numeric column and turns it into a numeric column.
    Inputs:
        col -> pd.Series: The pandas series that should be numeric
        replace_comma_with -> Literal['','.']: What commas will be 
                                               replaced with in the 
                                               would be numeric colum
    Outputs:
        col -> pd.Series: The should be numeric column now turned numeric

    """
    non_numeric = _find_not_numeric(col)
    col = [entry if entry not in non_numeric else np.nan for entry in col]
    col = [entry.strip().replace(',',replace_comma_with).replace(' ','') for entry in col]
    col = pd.to_numeric(col)
    return col

def _remove_accents(text: str) -> str:
    return ''.join(
        char for char in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(char)
    )

def _clean_column_name(column: str) -> str:
    column = _remove_accents(column)                                                                    
    column = (
        column.replace('.', ' ')
              .replace('?', ' ')
              .replace('¿', ' ')
              .replace('¡', ' ')
              .replace('!', ' ')
              .strip()
              .replace(' ', '_')
              .lower()
    )
    return column

def clean_df(data: pd.DataFrame,
             reindex_col: None|str = None,
             bool_cols: None|list[str]|str = None,
             true_pats: None|list[str]|str = None,
             num_cols: None|list[str]|str = None,
             replace_comma_with: Literal['','.']= '',
             drop_na: bool = False,
             drop_dupes: bool = False,
             drop_cols: None|list[str]|str = None,
             ) -> pd.DataFrame:
    """
    Cleans a pandas dataframe removing special charachters from column names and 
    changing the data types of specified columns.

    Inputs:
        data -> pd.DataFrame: The pandas dataframe that is going to be cleaned
        reindex_col -> None|string: If present is the column that was meant 
                                    to be the index contained in the dataframe
        bool_cols -> None|list[string]|string: If present it's either the column or columns
                                               that should be typed as booleans
        true_pats -> None|list[string]|string: Required if bool_cols are passed, it is the 
                                               pattern or patterns that are associated with
                                               a true boolean column (either one pattern in 
                                               general or one pattern per column)
        num_cols  -> None|list[string]|string: A set of columns or asingle columns that are 
                                               represented as objects but should be numeric
        replace_comma_with -> Literal['','.']: Does nothing if num_cols is None. Decides what 
                                               commas will be replaced with in the numeric 
                                               columns
        drop_na -> bool: If true ensures that missing values will be dropped from the records
        drop_dupes -> bool: If true esnsures that duplicate records will be dropped from the
                            dataframe.
        drop_cols -> None|list[string]|string: If present they are the names of the columns that 
                                               are to be dropped
    Outputs:
        data -> pd.DataFrame: The clean version of the dataframe that was passed
    """

    data = data.copy()

    if reindex_col:
        data.index = data[reindex_col]
        data = data.drop(columns=reindex_col)

    if type(bool_cols) == list:
        if type(true_pats) == str:
            for col in bool_cols:
                data[col]=data[col].str.contains(true_pats,na=False)
        elif type(true_pats) == list:
            for col, pat in zip(bool_cols,true_pats):
                data[col]=data[col].str.contains(pat,na=False)
        else:
            raise ValueError

    elif type(bool_cols) == str:
        assert(type(true_pats)==str)
        data[bool_cols] = data[bool_cols].str.contains(true_pats,na=False)

    else:
        if true_pats:
            warnings.warn('true_pats was passed but no bool_cols were passed. True_pats was not used')

    if type(num_cols)==list:
        for col in num_cols:
            data[col] = _turn_numeric(data[num_cols],replace_comma_with)
    elif type(num_cols)==str:
       data[num_cols] = _turn_numeric(data[num_cols],replace_comma_with)
    
    if drop_na:
        data = data.dropna()

    if drop_dupes:
        data = data.drop_duplicates()

    if drop_cols:
        data = data.drop(columns=drop_cols)
    
    data.columns = [_clean_column_name(column) for column in data.columns]
    
    return data