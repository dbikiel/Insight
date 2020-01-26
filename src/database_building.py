import pandas as pd
import numpy as np
import json
import os.path
import pathlib
from tqdm import notebook, trange
import unidecode

# Function to convert json to list
def json_to_dict(filename, directory):
    res = {'tmdb_id': '',
           'imdb_id': '',
           'title': '',
           'year': '',
           'budget': '',
           'genres': '',
           'overview': '',
           'popularity': '',
           'revenue': '',
           'vote_average': '',
           'vote_count': '',
           'keywords': '',
           'similar_tmdbId': '',
           'recommendations_tmdbId': '',
           'actors': '',
           'characters': '',
           'director': '',
           'producer': '',
           'writer': ''}

    f = directory / filename
    if not f.exists():
        return res

    with open(f, "rb") as file:
        data = file.read()
        r = json.loads(data)

    res['tmdb_id'] = r['id']  # tmdb id
    res['imdb_id'] = r['imdb_id']  # imdb id
    res['title'] = r['title']  # title
    res['year'] = r['release_date'][:4]  # year
    res['budget'] = r['budget']  # budget
    res['genres'] = '|'.join([i['name'] for i in r['genres']])  # genres
    res['overview'] = r['overview'].strip()
    res['popularity'] = r['popularity']
    res['revenue'] = r['revenue']
    res['vote_average'] = r['vote_average']
    res['vote_count'] = r['vote_count']
    res['keywords'] = '|'.join([i['name'] for i in r['keywords']['keywords']])
    res['similar_tmdbId'] = '|'.join([str(i['id']) for i in r['similar']['results']])
    # res['similar_title'] = '|'.join([str(i['title']) for i in r['similar']['results']])
    res['recommendations_tmdbId'] = '|'.join([str(i['id']) for i in r['recommendations']['results']])
    # res['recommendations_title'] = '|'.join([str(i['title']) for i in r['recommendations']['results']])

    # Now the credits
    cast = r['credits']['cast']
    n = min(10, len(cast))

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
movie = '710'
res = json_to_dict(movie + '.txt', directory_TMDB)
#df_overview = pd.DataFrame([json_to_dict(str(i)+ '.txt', directory_TMDB) for i in tmdbId_list])
#df_overview['movieId'] = ids_df.movieId
#df_overview

for i in res:
    print(unidecode.unidecode(str(res[i])))


