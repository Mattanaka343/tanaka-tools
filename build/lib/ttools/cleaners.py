"""Utilities for dataframe cleaning.

This module contains functions that aid in the cleaning of pandas 
dataframes and on the handling and correcting of data types
"""

import string 
import warnings
import unicodedata
import json

try:
    import ollama
except ModuleNotFoundError:
    ollama = None

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

def _rename_dataframe_columns(
    df: pd.DataFrame,
    model: str = "qwen3:4b",
    sample_size: int = 5,
) -> pd.DataFrame:
    """
    Use a local Ollama LLM to semantically rename DataFrame columns.

    The model receives:
        - Original column name
        - Pandas dtype
        - A small sample of values

    It returns a mapping from original column names to
    descriptive snake_case names.
    """

    if ollama is None:
        raise ImportError(
            "The 'ollama' package is required to use _rename_dataframe_columns. "
            "Install it with 'pip install ollama'."
        )

    column_info = []

    for column in df.columns:
        values = df[column].dropna().head(sample_size).tolist()

        column_info.append({
            "original_name": str(column),
            "dtype": str(df[column].dtype),
            "sample_values": [str(v) for v in values],
        })

    prompt = f"""
You are a data engineering assistant.

Your task is to infer what each DataFrame column represents and
assign it an appropriate Python/pandas column name.

IMPORTANT:
- Infer the semantic meaning of the column from its name, dtype,
  and sample values.
- Use concise but descriptive names.
- Use lowercase snake_case.
- Do not use spaces.
- Do not use punctuation.
- Do not unnecessarily abbreviate words.
- Preserve the original meaning.
- Do not invent information that cannot be inferred.
- If a column represents a measurement, make the unit clear when
  the unit is known.
- Prefer names such as `engagement_rate`, `followers_gained`,
  `publication_date`, `video_duration_seconds`, etc.
- Two columns should only receive the same name if they genuinely
  represent the same concept.
- Return ONLY valid JSON.

Here are the columns:

{json.dumps(column_info, indent=2)}

Return a JSON object where:
- each key is the original column name
- each value is the recommended new column name

Example:

{{
    "Total Number of Impressions": "impressions",
    "Engagement Rate (%)": "engagement_rate",
    "Followers Gained": "followers_gained"
}}
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format="json",
    )

    mapping = json.loads(response["message"]["content"])

    # Safety check: make sure every original column has a mapping
    missing = set(df.columns) - set(mapping.keys())

    if missing:
        raise ValueError(
            f"LLM did not provide names for columns: {missing}"
        )

    # Make sure the LLM didn't create duplicate names
    new_names = list(mapping.values())

    if len(new_names) != len(set(new_names)):
        raise ValueError(
            "LLM generated duplicate column names: "
            f"{new_names}"
        )

    return df.rename(columns=mapping)

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


def _clean_text_value(value: object) -> object:
    if pd.isna(value):
        return value

    text = str(value).strip()

    if text == "":
        return text

    return text[:1].upper() + text[1:].lower()


def clean_df(data: pd.DataFrame,
             reindex_col: None|str = None,
             bool_cols: None|list[str]|str = None,
             model: str|None = None,
             true_pats: None|list[str]|str = None,
             num_cols: None|list[str]|str = None,
             text_cols: None|list[str]|str = None,
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
        num_cols  -> None|list[string]|string: A set of columns or a single column that are 
                                               represented as objects but should be numeric
        text_cols -> None|list[string]|string: A set of text columns to strip surrounding whitespace
                                               and normalize to sentence case (first letter uppercase, rest lowercase)
        replace_comma_with -> Literal['','.']: Does nothing if num_cols is None. Decides what 
                                               commas will be replaced with in the numeric 
                                               columns
        drop_na -> bool: If true ensures that missing values will be dropped from the records
        drop_dupes -> bool: If true ensures that duplicate records will be dropped from the
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
    
    if text_cols is not None:
        if isinstance(text_cols, str):
            text_cols = [text_cols]

        for col in text_cols:
            if col in data.columns:
                data[col] = data[col].map(_clean_text_value)

    if drop_na:
        data = data.dropna()

    if drop_dupes:
        data = data.drop_duplicates()

    if drop_cols:
        data = data.drop(columns=drop_cols)
    

    if model is not None:
        data = _rename_dataframe_columns(data,model=model)

    else:
        data.columns = [_clean_column_name(col) for col in data.columns]
    
    return data