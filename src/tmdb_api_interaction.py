import pandas as pd
import json
from tqdm import tqdm
import requests
import pathlib


def request_from_tmdb(movie_id, directory, API_KEY):
    """
    :param movie_id: Id of the movie requested from The Movie Database
    :param directory: directory where to save the data
    :param API_KEY: The Movie Database API key
    :return: JSON file containing all the information requested in the API call
    """
    r = requests.get('https://api.themoviedb.org/3/movie/' + str(movie_id) +
                     '?api_key=' + API_KEY + '&language=en-US&append_to_response=keywords,similar,recommendations,credits')

    # If the requests is correct, saves the file to disk
    if r.status_code == 200:
        with open(filename, 'w') as outfile:
            json.dump(r.json(), outfile)


# Load the movie list and find the corresponding code in The Movie Database
path = pathlib.Path(__file__).parent.absolute()
ids_filename = path.parents[0] / "data/ml-25m/links.csv"
ids_df = pd.read_csv(ids_filename, dtype=str)
tmdb_id_list = list(ids_df['tmdbId'])
rawdata_directory = path.parents[0] / "rawData/TMDB-metadata-62K/"


##### TMDB API KEY #####
api_file = path.parents[0] / "src/tmdb_api_key.txt"
API_KEY = api_file.read_text()
#######################

# Opens a guest session
r = requests.get('https://api.themoviedb.org/3/authentication/guest_session/new?api_key=' + API_KEY)

# For each movie, pull data if there is no file in the directory
for movie_id in tqdm(tmdb_id_list):
    movie = movie_id + '.txt'
    filename = rawdata_directory / movie
    if not filename.exists():
        request_from_tmdb(movie_id, filename, API_KEY)