import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import requests

movies_filename = '/Users/dbikiel/PycharmProjects/MovieDatabase/ml-25m/movies.csv'
movies_df = pd.read_csv(movies_filename)
ids_filename = '/Users/dbikiel/PycharmProjects/MovieDatabase/ml-25m/links.csv'
ids_df = pd.read_csv(ids_filename, dtype=str)