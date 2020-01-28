import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec

directory = '/Users/dbikiel/Documents/Insight-Local/notebooks/'

#load model
model = Doc2Vec.load(directory + 'model_doc2vec_20120123')

#load data
data = pd.read_csv(open(directory + 'movielens_62K_titles_overviews_summaries.csv'))
over = pd.read_csv(open('/Users/dbikiel/PycharmProjects/Insight/data/TMDB-metadata-62K.csv'), encoding='utf-8')
overview = over.overview

title = pd.read_csv(directory + 'movie_title.csv')
movieId = title.movieId
title = title.title

#####
def find_me_something(model, movieId, title, overview, words, topn = 500, movies = False):
    # Given a list of words, find the most similar movie

    # Extract words
    word1 = words[0]
    word2 = words[1]

    # Search similarity
    if not movies:
        vec1 = model[word1]
        vec2 = model[word2]
    else:
        titles_only = [i.split('(') for i in title]
        titles_only = [i[0].strip() for i in titles_only]
        movie1 = titles_only.index(word1)
        movie2 = titles_only.index(word2)
        vec1 = model.docvecs[movie1]
        vec2 = model.docvecs[movie2]

    vec1_sim = model.docvecs.most_similar([vec1], topn=topn)
    vec2_sim = model.docvecs.most_similar([vec2], topn=topn)

    # Unpack vectors
    vec1_sim_doctags, vec1_sim_sim = zip(*vec1_sim)
    vec2_sim_doctags, vec2_sim_sim = zip(*vec2_sim)

    # Create a list
    INTERSECTION = set(vec1_sim_doctags).intersection(set(vec2_sim_doctags))

    if movies:
        if movie1 in INTERSECTION:
            INTERSECTION.remove(movie1)

        if movie2 in INTERSECTION:
            INTERSECTION.remove(movie2)

    results = {}
    for i in INTERSECTION:
        res = {}
        res['id'] = movieId[i]
        res['title'] = title[i]
        res['overview'] = overview[i]
        sim1 = vec1_sim_sim[vec1_sim_doctags.index(i)]
        sim2 = vec2_sim_sim[vec2_sim_doctags.index(i)]

        res['user1_sim'] = np.round(sim1, 3)
        res['user2_sim'] = np.round(sim2, 3)
        res['mean_similarity'] = np.round(0.5 * (sim1 + sim2), 3)
        res['std_similarity'] = np.round(np.std(np.array([np.round(sim1, 3), np.round(sim2, 3)])), 3)
        res['inequality'] = np.round(sim1 - sim2, 3)
        res['product'] = np.round(sim1 * sim2, 3)

        results[i] = res

    return results

######

st.title('Ninjas versus Puppies')
selection = st.radio("Want titles or words?",('Movie Titles', 'Words'))
word1 = st.text_input('What do you want to see?')
word2 = st.text_input('And what do you want to see?')
st.write('You want to see something with ', word1, ' and ', word2)
st.write('Behold!')

if selection == 'Words':
    movies = False
else:
    movies = True

try:
    results = find_me_something(model, movieId, title, overview, [word1, word2], 10000, movies)
    res = pd.DataFrame(results)
    res = res.transpose()
    final = res.sort_values(by=['mean_similarity'], ascending=False).head(10)
    #final
    st.table(final[['title', 'overview', 'mean_similarity', 'user1_sim','user2_sim']])
except:
    st.write('You want to see something different, maybe?')
