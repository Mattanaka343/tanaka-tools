"""Utilities for data extraction from different file types

This module provides functions that aids in the extraction of data from different files that are
typically not supported by pndas's read functions.
"""

import pandas as pd
import xml.etree.ElementTree as ET

def read_excel_xml(path:str) -> pd.DataFrame:
    """
    Reads excel files older than the 2003 version of excel via decoding of the xml layers

    Input:
        path -> string: The relative path to the xml file
    
    Output:
        df -> pd.Dataframe: The pandas data frame containingall the information that was extracted
                             from the old excel file.
              
    """
    root = ET.parse(path).getroot()
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}

    rows = []
    for row in root.findall(".//ss:Worksheet/ss:Table/ss:Row", ns):
        rows.append([
            cell.findtext("ss:Data", default="", namespaces=ns)
            for cell in row.findall("ss:Cell", ns)
        ])

    return pd.DataFrame(rows[1:], columns=rows[0])

