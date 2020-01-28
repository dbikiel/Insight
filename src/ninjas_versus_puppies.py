import streamlit as st
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from search_similar_title import *


#load model
model = Doc2Vec.load('../models/model_doc2vec_20120123')
#print('Done!')

#load data
over = pd.read_csv('../data/TMDB-metadata-62K.csv')
overview = over.overview

title = pd.read_csv('../data/movies.csv')
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
        #titles_only = [i.split('(') for i in title]
        #titles_only = [i[0].strip() for i in titles_only]
        #movie1 = titles_only.index(word1)
        #movie2 = titles_only.index(word2)
        movie1 = word1
        movie2 = word2
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
w1 = st.empty()
w2 = st.empty()
word1 = st.empty()
word2 = st.empty()
word1_list = st.empty()
word2_list = st.empty()


if selection == 'Words':
    movies = False
else:

    movies = True

    w1 = st.text_input('What do you want to see?')
    res = search_similar_title(w1, title, 10)
    ids1, word1_list = zip(*res)
    word1 = st.selectbox('Please select:', word1_list)


    w2 = st.text_input('And what do you want to see?')
    res = search_similar_title(w2, title, 10)
    ids2, word2_list = zip(*res)
    word2 = st.selectbox('Please select:', word2_list)

    st.write('You want to see something close to ', word1, ' and ', word2)

    word1 = int(ids1[word1_list.index(word1)])
    word2 = int(ids2[word2_list.index(word2)])
    #print(word1)
    #print(word2)

try:
    results = find_me_something(model, movieId, title, overview, [word1, word2], 5000, movies)
    res = pd.DataFrame(results)
    res = res.transpose()
    final = res.sort_values(by=['mean_similarity'], ascending=False).head(5)
    #final
    st.table(final[['title', 'overview', 'mean_similarity']])
except:
    st.write('You want to see something different, maybe?')
