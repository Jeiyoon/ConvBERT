import sys
import csv

csv.field_size_limit(sys.maxsize)

import pandas as pd

def csv_to_text(in_file: str, out_file: str) -> None:
    SEPARATOR = u"\u241D"
    df = pd.read_csv(in_file, sep=SEPARATOR, engine="python")

    with open(out_file, "w") as f:
        for index, row in df.iterrows():
            f.write(row["text"]) # title 과 text를 중복 되므로 text만 저장 함
            f.write("\n\n\n\n") # 구분자
