import pandas as pd
import json
import pathlib
from tqdm import tqdm
import unidecode
import csv
import re

def json_to_dict(filename, directory):
    """
    Function to parse the json file pulled from TMDB to a dictionary containing metadata (including the overview)
    :param filename:
    :param directory:
    :return:
    """
    # Creates the empty dictionary to populate
    res = {'tmdb_id': '', 'imdb_id': '', 'title': '', 'year': '',
           'budget': '', 'genres': '', 'overview': '', 'popularity': '', 'revenue': '', 'vote_average': '',
           'vote_count': '', 'keywords': '', 'similar_tmdbId': '','recommendations_tmdbId': '', 'actors': '',
           'characters': '', 'director': '', 'producer': '', 'writer': ''}

    # If the fle does not exist returns an empty dictionary
    f = directory / filename
    if not f.exists():
        return res

    # Open and reads the json file
    with open(f, "rb") as file:
        data = file.read()
        r = json.loads(data)

    # Read metadata and overview
    res['tmdb_id'] = r['id']  # tmdb id
    res['imdb_id'] = r['imdb_id']  # imdb id
    res['title'] = r['title']  # title
    res['year'] = r['release_date'][:4]  # year
    res['budget'] = r['budget']  # budget
    res['genres'] = '|'.join([i['name'] for i in r['genres']])  # genres
    res['overview'] = r['overview'].strip()

    ####
    # Remove special characters from the overview to avoid problems later during saving and reading the data
    final_text = res['overview']
    review_text = re.sub(r"[^A-Za-z0-9(),!.?\'`]", " ", final_text)
    #review_text = re.sub(r"\'s", " 's ", final_text)
    #review_text = re.sub(r"\'ve", " 've ", final_text)
    #review_text = re.sub(r"n\'t", " 't ", final_text)
    #review_text = re.sub(r"\'re", " 're ", final_text)
    #review_text = re.sub(r"\'d", " 'd ", final_text)
    #review_text = re.sub(r"\'ll", " 'll ", final_text)
    #review_text = re.sub(r",", " ", final_text)
    #review_text = re.sub(r"\.", " ", final_text)
    #review_text = re.sub(r"!", " ", final_text)
    #review_text = re.sub(r"\(", " ( ", final_text)
    #review_text = re.sub(r"\)", " ) ", final_text)
    #review_text = re.sub(r"\?", " ", final_text)
    #review_text = re.sub(r"\s{2,}", " ", final_text)
    res['overview'] = review_text
    #####

    res['popularity'] = r['popularity']
    res['revenue'] = r['revenue']
    res['vote_average'] = r['vote_average']
    res['vote_count'] = r['vote_count']
    res['keywords'] = '|'.join([i['name'] for i in r['keywords']['keywords']])
    res['similar_tmdbId'] = '|'.join([str(i['id']) for i in r['similar']['results']])
    # res['similar_title'] = '|'.join([str(i['title']) for i in r['similar']['results']])
    res['recommendations_tmdbId'] = '|'.join([str(i['id']) for i in r['recommendations']['results']])
    # res['recommendations_title'] = '|'.join([str(i['title']) for i in r['recommendations']['results']])

    # Now extract info from the credits
    cast = r['credits']['cast']
    n = min(10, len(cast))

    # Actors and Characters
    actors = []
    characters = []
    for i in range(n):
        actors.append(r['credits']['cast'][i]['name'])
        characters.append(r['credits']['cast'][i]['character'])

    res['actors'] = '|'.join(actors)
    res['characters'] = '|'.join(characters)

    # Director, Executive Producer and Writer
    director = []
    producer = []
    writer = []

    for i in r['credits']['crew']:
        if i['job'] == 'Director':
            director.append(i['name'])
        elif i['job'] == 'Executive Producer':
            producer.append(i['name'])
        elif i['job'] == 'Writer':
            writer.append(i['name'])

    res['director'] = '|'.join(director)
    res['producer'] = '|'.join(producer)
    res['writer'] = '|'.join(writer)

    # Cleaning names with non english characters
    for key in res:
        res[key] = unidecode.unidecode(str(res[key]))

    return res

# Loading files
path = pathlib.Path(__file__).parent.absolute()

ids_filename = path.parents[0] / "data/ml-25m/links.csv"
ids_df = pd.read_csv(ids_filename, dtype=str)

movies_filename = path.parents[0] / "data/ml-25m/movies.csv"
movies_df = pd.read_csv(movies_filename, dtype=str)

tmdb_id_list = list(ids_df['tmdbId'])
rawdata_directory = path.parents[0] / "rawData/TMDB-metadata-62K/"

# Execute
directory_TMDB = path.parents[0] / "rawData/TMDB-metadata-62K"

#Saves the database to disk
with open(path.parents[0] / "data/TMDB-metadata-62K.csv", mode='w') as file:
    file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = False
    for i in tqdm(tmdb_id_list):
        tmp = json_to_dict(str(i) + '.txt', directory_TMDB)
        if not header:
            h = tmp.keys()
            file_writer.writerow(h)
            header = True
        file_writer.writerow(tmp.values())
