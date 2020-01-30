import streamlit as st
import pandas as pd
import numpy as np
import random

from gensim.models.doc2vec import Doc2Vec

from bokeh.plotting import figure, output_notebook, show, ColumnDataSource, output_file
from bokeh.plotting import figure

import warnings
import difflib
from PIL import Image

warnings.filterwarnings("ignore")

####### helping functions #######

def find_middle_with_filter(model, data_df, movieid_to_doctags, movies, topn=2000):
    movie1 = int(movies[0])
    movie2 = int(movies[1])

    # titles and movieIds from the dataframe
    title = list(data_df['title_year'])
    movieId = list(data_df['movieId'])

    movieid_to_title = {movie: tit for movie, tit in zip(movieId, title)}
    # set of vectors to search the similarity
    vectors = set([movieid_to_doctags[movie] for movie in movieId])

    # dict to convert doctags
    doctag_to_movieId = {j: i for i, j in zip(movieid_to_doctags.keys(), movieid_to_doctags.values())}
    doctag_to_title = {j: i for i, j in zip(vectors, title)}

    vec1 = model.docvecs[movieid_to_doctags[movie1]]
    vec2 = model.docvecs[movieid_to_doctags[movie2]]

    vec1_sim = model.docvecs.most_similar([vec1], topn=topn)
    vec2_sim = model.docvecs.most_similar([vec2], topn=topn)

    # Unpack vectors
    vec1_sim_doctags, vec1_sim_sim = zip(*vec1_sim)
    vec2_sim_doctags, vec2_sim_sim = zip(*vec2_sim)

    # Create a list
    INTERSECTION = set(vec1_sim_doctags).intersection(set(vec2_sim_doctags))
    INTERSECTION = INTERSECTION.intersection(vectors)

    if movie1 in INTERSECTION:
        INTERSECTION.remove(movie1)

    if movie2 in INTERSECTION:
        INTERSECTION.remove(movie2)

    movie_id = []
    user1_sim = []
    user2_sim = []
    mean_sim = []
    std_sim = []
    ineq_sim = []
    prod_sim = []
    mean_ineq_sim = []
    std_over_mean_sim = []
    for i in INTERSECTION:
        movie_id.append(doctag_to_movieId[i])
        # titles.append(movieid_to_title[doctag_to_movieId[i]])

        sim1 = vec1_sim_sim[vec1_sim_doctags.index(i)]
        sim2 = vec2_sim_sim[vec2_sim_doctags.index(i)]

        user1_sim.append(np.round(sim1, 3))
        user2_sim.append(np.round(sim2, 3))
        mean_sim.append(np.round(0.5 * (sim1 + sim2), 3))
        std_sim.append(np.round(np.std(np.array([sim1, sim2])), 3))
        ineq_sim.append(np.round(abs(sim1 - sim2), 3))
        prod_sim.append(np.round(sim1 * sim2, 3))
        mean_ineq_sim.append(np.round(0.5 * (sim1 + sim2) - 0.5 * abs(sim1 - sim2), 3))
        std_over_mean_sim.append(np.round(np.std(np.array([sim1, sim2])) / (0.5 * (sim1 + sim2)), 3))

    #    results = {'id': movie_id, 'title': titles, 'user1_sim': user1_sim,
    results = {'id': movie_id, 'user1_sim': user1_sim,
               'user2_sim': user2_sim, 'mean_sim': mean_sim, 'std_sim': std_sim, 'ineq_sim': ineq_sim,
               'prod_sim': prod_sim, 'mean_ineq_sim': mean_ineq_sim, 'std_over_mean_sim': std_over_mean_sim}

    res = pd.DataFrame(results)
    final = res.sort_values(by=['mean_ineq_sim'], ascending=False).head(6)
    # print(final)
    # return  final[['id','title', 'mean_sim','ineq_sim','mean_ineq_sim']]
    # return list(final['id'])[0]
    return list(final['id'])


def make_slider_with_filter(model, data_df, movieid_to_doctags, movie_ini, movie_end, topn=2000):
    movie_0 = movie_ini
    movie_8 = movie_end

    # 4
    # list_average = find_middle(model, title, movieId, [movie_0, movie_8], topn = topn)
    list_average = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_0, movie_8], topn=topn)
    movie_4 = random.choice(list_average)

    # 2
    list_ninjas_1 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_0, movie_4], topn=topn)
    movie_2 = random.choice(list_ninjas_1)

    # 6
    list_puppies_1 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_4, movie_8], topn=topn)
    movie_6 = random.choice(list_puppies_1)

    # 1
    list_ninjas_2 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_0, movie_2], topn=topn)
    movie_1 = random.choice(list_ninjas_2)

    # 5
    list_puppies_2 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_4, movie_6], topn=topn)
    movie_5 = random.choice(list_puppies_2)

    # 3
    list_ninjas_3 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_2, movie_4], topn=topn)
    movie_3 = random.choice(list_ninjas_3)

    # 7
    list_puppies_3 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_6, movie_8], topn=topn)
    movie_7 = random.choice(list_puppies_3)

    # Return list of lists

    res = [list_ninjas_2, list_ninjas_1, list_ninjas_3, list_average,
           list_puppies_1, list_puppies_2, list_puppies_3]
    return res


def movie_filter(movies_df, min_votes=0, min_average=0):
    return movies_df[(movies_df.vote_average >= int(min_average)) & (movies_df.vote_count >= int(min_votes))]


def search_similar_title(pattern, titles, n_results, threshold=0.4):
    """
    Search closest title from pattern
    :param pattern: string or movie
    :param titles: dataframe with title list
    :param n_results: number of results suggested
    :param threshold: from difflib
    :return: list of potential titles
    """
    title_list = list(titles.values)
    res = difflib.get_close_matches(pattern, title_list, n_results, threshold)
    res.sort()
    best = []
    for i in res:
        if i in title_list:
            best.append((title_list.index(i), i))
    return best


##################################
st.title('Ninjas versus Puppies')

# We import seaborn to make nice plots.
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

plot = st.sidebar.checkbox('Want to see the plot?', value=False, key=None)

if plot:
    ########
    data_for_bokeh = pd.read_csv('../data/pds_data_df_62K.csv', sep='\t')
    source = ColumnDataSource(data_for_bokeh)
    TOOLTIPS = [("Title", "@title")]
    plotting_movies = figure(title="Movies' space", tooltips=TOOLTIPS, plot_width=300, plot_height=300)
    plotting_movies.circle('x_val', 'y_val', color='black', alpha=0.01, source=source)
    plotting_movies.toolbar.logo = None
    plotting_movies.toolbar_location = None
    plotting_movies.xgrid.grid_line_color = None
    plotting_movies.ygrid.grid_line_color = None
    plotting_movies.axis.visible = False
    st.bokeh_chart(plotting_movies)
    # show(plotting_comments)
    #########


@st.cache
def load_data():
    model1 = Doc2Vec.load('../models/model_doc2vec_20120123')
    full_df = pd.read_csv('../data/TMDB-metadata-62K.csv')
    titles1 = pd.read_csv('../data/movies.csv')

    full_df['title_year'] = list(titles1.title)
    full_df['movieId'] = list(titles1.movieId)
    full_df = full_df[['movieId', 'title_year', 'overview', 'genres', 'vote_average', 'vote_count']]

    return [model1, full_df, titles1]


# load model and data
model, data_df, titles = load_data()

# Sidebar
st.sidebar.markdown('What do you want to see?')

#movie1 = st.sidebar.text_input('Choose movieId for 1:')
w1 = st.sidebar.text_input('Choose your first movie!')
res = search_similar_title(w1, data_df.title_year, 20)
ids1, word1_list = zip(*res)
title1 = st.sidebar.selectbox('Please select:', word1_list)
movie1 = data_df[data_df['title_year'] == title1]['movieId']


#movie2 = st.sidebar.text_input('Choose movieId for 2:')
w2 = st.sidebar.text_input('Choose your second movie!')
res = search_similar_title(w2, data_df.title_year, 20)
ids2, word2_list = zip(*res)
title2 = st.sidebar.selectbox('Please select:', word2_list)
movie2 = data_df[data_df['title_year'] == title2]['movieId']

movieid_to_doctags = {movie: i for i, movie in enumerate(titles.movieId)}

min_average = st.sidebar.slider('Min Rating?', min_value=0.0, max_value=10.0, value=5.0, step=0.5, format=None, key = None)
min_votes = st.sidebar.slider('Min Popularity?', min_value=0, max_value=1000, value=500, step=100, format=None, key = None)

submit = st.sidebar.button('Submit')
res_value = st.sidebar.slider('More ninjas or puppies?', min_value=0, max_value=6, value=3, step=1, format=None, key = None)

# if submit:
data = data_df.copy()
popular_movies = movie_filter(data, min_votes, min_average)
res = make_slider_with_filter(model, popular_movies, movieid_to_doctags, movie1, movie2, 5000)

# Make the Images!
image_list = []
for i in res[res_value]:
    image = Image.open('../rawData/pictures/' + str(i) + '.jpg')
    image_list.append(image)
st.image(image_list)
