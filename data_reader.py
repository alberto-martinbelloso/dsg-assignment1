import pandas as pd
import json


def get_data(file):
    path = './data/' + file
    if file == "fifa.csv" or file == "telecom_churn.csv":
        data = pd.read_csv(path, sep=",", index_col=False)
    else:
        with open(path) as f:
            data = json.load(f)
        # data = pd.read_json(path)

    return data
