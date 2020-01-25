from requests import get
from bs4 import BeautifulSoup
from bleach.sanitizer import Cleaner
import pandas as pd
from tqdm import tqdm
import pathlib


def scrap_and_clean(movie_id):
    """
    Scraps the plot and synopsis information from a particular movie in the IMDB

    :param movie_id: int corresponding to a particular movie in the IMDB
    :return: A list containing the plot and synopsis
    """
    # Scraps from a page at imdb
    url = 'https://www.imdb.com/title/' + str(movie_id) + '/plotsummary?ref_=tt_stry_pl#synopsis'
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')

    # Get the cointainers with the summary plots and the synopsis
    movie_containers = html_soup.find_all('li', class_='ipl-zebra-list__item')

    # creates a Cleaner to prettify the text
    cleaner = Cleaner(strip=True, tags=[])

    # Creates a list and put everything each plot in a list
    texts = []
    for text in movie_containers:
        sanitized = cleaner.clean(str(text))
        texts.append(sanitized.split('\n\n')[0])

    return texts


# Read the imdb ids from the movielens database file
path = pathlib.Path(__file__).parent.absolute()
filename = path.parents[0] / "data/ml-25m/links.csv"

# generates a list
ids_df = pd.read_csv(filename, dtype=str)
imdb_id_list = ids_df['imdbId'].values

# Directory to save the scraped data
rawdata_directory = path.parents[0] / "rawData/IMDB-plots-62K/"

# Scrapes for each file in the imdb list
for movie_id in tqdm(imdb_id_list):
    movie = 'tt' + movie_id + '.txt'
    filename = rawdata_directory / movie

    # If it does not exists, scraps
    if not filename.exists():
        plot = scrap_and_clean('tt' + movie_id)
        with open(filename, 'w') as f:
            for p in plot:
                f.write("%s\n" % p)
