import pandas as pd
import json
import pathlib
from tqdm import tqdm
import unidecode
import csv
import re

def clean_string(p):
    p = unidecode.unidecode(p.strip())
    p = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", p)
    p = re.sub(r"\'", "'", p)
    return p

# Loading files
path = pathlib.Path(__file__).parent.absolute()

ids_filename = path.parents[0] / "data/ml-25m/links.csv"
ids_df = pd.read_csv(ids_filename, dtype=str)

movies_filename = path.parents[0] / "data/ml-25m/movies.csv"
movies_df = pd.read_csv(movies_filename, dtype=str)

imdb_id_list = list(ids_df['imdbId'])
movie_id_list = list(ids_df['movieId'])
rawdata_directory = path.parents[0] / "rawData/IMDB-plots-62K/"

# Put everything in one file
with open(path.parents[0] / "data/IMDB-summaries-62K.csv", mode='w') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['imdbId','summaries'])
    for movie in tqdm(imdb_id_list):
        plots = []
        try:
            movie_file = 'tt' + str(movie) + '.txt'
            with open( rawdata_directory / movie_file,'r') as m:
                plot = list(m)
                plot = [clean_string(p) for p in plot]
        except:
            plot = []
        summary = ''.join(plot)
        file_writer.writerow([movie,summary])