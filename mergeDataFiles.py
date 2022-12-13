import os
import glob
import pandas as pd

path = "/Users/lucie/PycharmProjects/projectTest"
all_files = glob.glob(os.path.join(path, "Coordinates_*.csv"))

df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged  = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "merged.csv")