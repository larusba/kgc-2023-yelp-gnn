import json
import pandas as pd

from utils import flat_object

def get_table(path):
    list_of_dicts = []
    with open(path, 'r') as f:
        for line in f:
            list_of_dicts.append(flat_object(json.loads(line)))
    df = pd.DataFrame.from_dict(list_of_dicts).dropna(axis=1, how='all')
    return df
