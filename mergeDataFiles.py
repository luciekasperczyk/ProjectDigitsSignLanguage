"""
'MergeDataFiles' FILE
Subject : module project in Computer Science
Semester : F2022
Authors :
    - Nasrdin Ahmed Aden
    - Zainab Ahmad
    - Lucie Kasperczyk
    - Hermann Yunus Knudsen
Code inspirations :
- Libraries :
    * glob library : https://docs.python.org/3/library/glob.html
    * os library : https://docs.python.org/3/library/os.html
- Merge CSV files :
    * read file : https://www.w3schools.com/python/pandas/pandas_csv.asp
    * concatenation : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
"""

import os
import glob
import pandas as pd

path = "/Users/lucie/PycharmProjects/projectTest"
all_files = glob.glob(os.path.join(path, "Coordinates_*.csv"))

df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged  = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "merged.csv")