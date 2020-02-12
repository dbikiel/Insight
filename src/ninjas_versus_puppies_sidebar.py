import streamlit as st
import pandas as pd
import numpy as np
import random
from gensim.models.doc2vec import Doc2Vec
from bokeh.plotting import figure, ColumnDataSource
import seaborn as sns
from PIL import Image
import time
import difflib

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
import warnings

warnings.filterwarnings("ignore")


####### helping functions #######
@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def find_middle_with_filter(model, data_df, movieid_to_doctags, movies, topn, remove_movies):
    """
    Function to search a movie in between two other movies. Uses the a doc2vec trained model an a database
    of movies to find the topn closest movies.
    :param model: a doc2vec model
    :param data_df: a pandas dataframe containing id, title, rating, votes, overview
    :param movieid_to_doctags: dict to transform movie id to doctag id
    :param movies: list of id of 2 movies
    :param topn: number of similarity searches for each movie integer
    :param remove_movies: id of movies to remove from the final result
    :return:
    """

    # get the ids from the list
    movie1 = int(movies[0])
    movie2 = int(movies[1])

    to_remove = [int(remove_movies[0]), int(remove_movies[1])]

    # titles and movieIds from the dataframe
    title = list(data_df['title_year'])
    movieId = list(data_df['movieId'])

    movieid_to_title = {movie: tit for movie, tit in zip(movieId, title)}

    # set of vectors to search the similarity
    vectors = set([movieid_to_doctags[movie] for movie in movieId])

    # dict to convert doctags
    doctag_to_movieId = {j: i for i, j in zip(movieid_to_doctags.keys(), movieid_to_doctags.values())}
    #doctag_to_title = {j: i for i, j in zip(vectors, title)}

    # Get the vectors of the target movies
    vec1 = model.docvecs[movieid_to_doctags[movie1]]
    vec2 = model.docvecs[movieid_to_doctags[movie2]]

    # Search topn similar movies for each movie
    vec1_sim = model.docvecs.most_similar([vec1], topn=topn)
    vec2_sim = model.docvecs.most_similar([vec2], topn=topn)

    # Unpack vectors
    vec1_sim_doctags, vec1_sim_sim = zip(*vec1_sim)
    vec2_sim_doctags, vec2_sim_sim = zip(*vec2_sim)

    # Create the intersection of the most similar movies
    intersection = set(vec1_sim_doctags).intersection(set(vec2_sim_doctags))
    intersection = intersection.intersection(vectors)

    # if the target movies are in the intersection, is removed
    if movieid_to_doctags[movie1] in intersection:
        intersection.remove(movieid_to_doctags[movie1])
    if movieid_to_doctags[movie2] in intersection:
        intersection.remove(movieid_to_doctags[movie2])

    # Remove the movies in the remove list if they are in the intersection
    for movie in to_remove:
        if movieid_to_doctags[movie] in intersection:
            intersection.remove(movieid_to_doctags[movie])

    # Create list for different parameters to produce the final result
    # id
    movie_id = []

    # similarity to movie 1 or 2
    user1_sim = []
    user2_sim = []

    # mean and std similarity
    mean_sim = []
    std_sim = []

    # absolute difference of the difference between the similarity
    ineq_sim = []

    # Product of the similarity
    prod_sim = []

    # mean similarity - absolute difference
    mean_ineq_sim = []

    # std divided by mean similarity
    std_over_mean_sim = []

    # For each movie found calculates all the parameters
    for i in intersection:

        #id
        movie_id.append(doctag_to_movieId[i])

        # movie similarity versus movie1 and movie2
        sim1 = vec1_sim_sim[vec1_sim_doctags.index(i)]
        sim2 = vec2_sim_sim[vec2_sim_doctags.index(i)]

        # similarities
        user1_sim.append(np.round(sim1, 3))
        user2_sim.append(np.round(sim2, 3))

        # mean and std
        mean_sim.append(np.round(0.5 * (sim1 + sim2), 3))
        std_sim.append(np.round(np.std(np.array([sim1, sim2])), 3))

        # absolute difference, product, mean - abs, std / mean
        ineq_sim.append(np.round(abs(sim1 - sim2), 3))
        prod_sim.append(np.round(sim1 * sim2, 3))
        mean_ineq_sim.append(np.round(0.5 * (sim1 + sim2) - 0.5 * abs(sim1 - sim2), 3))
        std_over_mean_sim.append(np.round(np.std(np.array([sim1, sim2])) / (0.5 * (sim1 + sim2)), 3))

    # creates a dataframe with all the results
    results = {'id': movie_id, 'user1_sim': user1_sim,
               'user2_sim': user2_sim, 'mean_sim': mean_sim, 'std_sim': std_sim, 'ineq_sim': ineq_sim,
               'prod_sim': prod_sim, 'mean_ineq_sim': mean_ineq_sim, 'std_over_mean_sim': std_over_mean_sim}

    res = pd.DataFrame(results)

    # produces the final result, sorting by mean - abs. Only first 6 results
    final = res.sort_values(by=['mean_ineq_sim'], ascending=False).head(6)
    print(final[['user1_sim','user2_sim']])
    # returns a list of movie ids
    return list(final['id'])


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def movie_similarity(model, movieid_to_doctags, movies):
    """
    Function to calculate similarity between movies
    :param model: a doc2vec model
    :param movieid_to_doctags: dict to transform movie id to doctag id
    :param movies: list of id of 2 movies
    :return: similarity
    """

    # get the ids from the list
    movie1 = int(movies[0])
    movie2 = int(movies[1])

    sim = model.docvecs.similarity(movieid_to_doctags[movie1], movieid_to_doctags[movie2])
    return sim



@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def make_slider_with_filter(model, data_df, movieid_to_doctags, movie_ini, movie_end, topn):
    """
    Builds the list of list of similar movies to be explored by a slider. It builds sequentially by searching the middle
    point between movie_ini and movie_end (7 list in total)

    :param model: doc2vec
    :param data_df: dataframe with movie id, overview, title
    :param movieid_to_doctags: dict to transform movie id to doctag
    :param movie_ini: target movie on the left
    :param movie_end: target movie on the right
    :param topn: search the first topn similar items
    :return: a list of list of (at most) movie ids in the middle
    """

    # Extreme points
    movie_0 = movie_ini
    movie_8 = movie_end
    top = topn

    # List 4 (middle one)
    list_average = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_0, movie_8], top,
                                           [movie_0, movie_8])

    #Select a movie at random to be the extreme of the next list
    movie_4 = random.choice(list_average)
    # movie_4 = list_average[0]

    # List 2 (middle between left and middle)
    list_ninjas_1 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_0, movie_4], top,
                                            [movie_0, movie_8])

    # Select a movie at random to be the extreme of the next list
    movie_2 = random.choice(list_ninjas_1)
    # movie_2 = list_ninjas_1[0]

    # List 6 (middle between right and middle movie)
    list_puppies_1 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_4, movie_8], top,
                                             [movie_0, movie_8])

    # Select a movie at random to be the extreme of the next list
    movie_6 = random.choice(list_puppies_1)
    # movie_6 = list_puppies_1[0]

    # List 1 (first list to the right of movie ini)
    list_ninjas_2 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_0, movie_2], top,
                                            [movie_0, movie_8])

    # Select a movie at random to be the extreme of the next list
    movie_1 = random.choice(list_ninjas_2)
    # movie_1 = list_ninjas_2[0]

    # List 7 (first to the left from movie end)
    list_puppies_3 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_6, movie_8], top,
                                             [movie_0, movie_8])

    # Select a movie at random to be the extreme of the next list
    movie_7 = random.choice(list_puppies_3)
    # movie_7 = list_puppies_3[0]

    # List 5 (first list to the right of the middle list)
    list_puppies_2 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_4, movie_6], top,
                                             [movie_0, movie_8])

    # Select a movie at random to be the extreme of the next list
    movie_5 = random.choice(list_puppies_2)
    # movie_5 = list_puppies_2[0]

    # List 3 (first list to the right of the middle list)
    list_ninjas_3 = find_middle_with_filter(model, data_df, movieid_to_doctags, [movie_2, movie_4], top,
                                            [movie_0, movie_8])

    # Select a movie at random to be the extreme of the next list
    movie_3 = random.choice(list_ninjas_3)
    # movie_3 = list_ninjas_3[0]

    # Return list of lists
    res = [list_ninjas_2, list_ninjas_1, list_ninjas_3, list_average,
           list_puppies_1, list_puppies_2, list_puppies_3]
    return res


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def movie_filter(movies_df, minvotes, minaverage):
    """
    Filters the original data frame to
    :param movies_df: original data
    :param minvotes: minimum votes kept
    :param minaverage: min rating kept
    :return: dataframe of movies filtered
    """
    filtered = movies_df[(movies_df.vote_average >= int(minaverage)) & (movies_df.vote_count >= int(minvotes))]
    #print(filtered)
    return filtered

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def search_similar_title(pattern, titles, n_results, threshold=0.4):
    """
    Search closest title from pattern
    :param pattern: string or movie
    :param titles: dataframe with title list
    :param n_results: number of results suggested
    :param threshold: from difflib
    :return: list of potential titles
    """

    # List of titles
    title_list = list(titles.values)

    # Result of the search pattern
    res = difflib.get_close_matches(pattern, title_list, n_results, threshold)
    res.sort()

    # Makes a tuple for each title and movie id with close match to the pattern
    best = []
    for i in res:
        if i in title_list:
            best.append((title_list.index(i), i))
    return best


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def plot_bokeh():
#def plot_bokeh(m1, m2, res, res_value):
    """
    Plots the Bert embedding of the overviews for th 62K movies. Can mark particular movies in the diagram
    :param m1: movie1
    :param m2: movie2
    :param res: list of list of movies from the slider
    :param res_value: value of the slider
    :return: plot of embeddings
    """

    # Reads data
    data_for_bokeh = pd.read_csv('../data/pds_data_df_62K.csv', sep='\t')
    source = ColumnDataSource(data_for_bokeh)

    # Format of the tooltips to show title
    TOOLTIPS = [("", "@title")]

    # size
    plotting_movies = figure(tooltips=TOOLTIPS, plot_width=800, plot_height=800)

    # Plot
    plotting_movies.circle('x_val', 'y_val', color='blue', alpha=0.05, source=source)

    # Remove axis, grids, box
    plotting_movies.toolbar.logo = None
    plotting_movies.toolbar_location = None
    plotting_movies.xgrid.grid_line_color = None
    plotting_movies.ygrid.grid_line_color = None
    plotting_movies.axis.visible = False
    plotting_movies.outline_line_color = "white"

    # Plot the movie points
    # if len(m1)> 0:
    #     mark_movies = st.sidebar.checkbox('Want to mark the movies?', value=False, key=None)
    #
    #     if mark_movies:
    #         m1_in_bokeh = movieid_to_doctags[list(m1)[0]]
    #         m2_in_bokeh = movieid_to_doctags[list(m2)[0]]
    #         plotting_movies.circle(data_for_bokeh.loc[m1_in_bokeh].x_val, data_for_bokeh.loc[m1_in_bokeh].y_val,
    #                                color='green', alpha=0.75, size=15)
    #         plotting_movies.circle(data_for_bokeh.loc[m2_in_bokeh].x_val, data_for_bokeh.loc[m2_in_bokeh].y_val,
    #                                color='red', alpha=0.75, size=15)
    #
    #         for i in res[res_value]:
    #             plotting_movies.circle(data_for_bokeh.loc[movieid_to_doctags[i]].x_val,
    #                                    data_for_bokeh.loc[movieid_to_doctags[i]].y_val,
    #                                    color='yellow', alpha=0.75, size=15)
    return plotting_movies


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def put_target_images(movie1, movie2):
    """
    Function to create list of images for the target movies
    :param movie1: movie1
    :param movie2: movie2
    :return: list of images
    """
    target_image_list = []

    # Create a list containing the movies and a plus in the middle
    for i in [list(movie1)[0], 'plus', list(movie2)[0]]:
        # create a list of images from the saved ones
        if i == 'plus':
            image = Image.open('../rawData/pictures/plus_icon.png')
        else:
            try:
                image = Image.open('../rawData/pictures/' + str(i) + '.jpg')
            except:
                image = Image.open('../rawData/pictures/blank.jpg')
        target_image_list.append(image)

    # return list of images
    return target_image_list


@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def make_images(res, res_value):
    """
    Makes the list of the images for each list in the slider
    :param res: list of movies
    :param res_value: value of the slider
    :return: list of images
    """
    # Make the list of images
    image_list = []
    if res_value is not None:
        for i in res[res_value]:
            try:
                image = Image.open('../rawData/pictures/' + str(i) + '.jpg')
            except:
                image = Image.open('../rawData/pictures/blank.jpg')
            image_list.append(image)
    else:
        for i in res:
            try:
                image = Image.open('../rawData/pictures/' + str(i) + '.jpg')
            except:
                image = Image.open('../rawData/pictures/blank.jpg')
            image_list.append(image)
    return image_list


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data():
    """
    Loads the model and the dataframe with the data
    :return: dataframes and model
    """
    model1 = Doc2Vec.load('../models/model_doc2vec_20120123')
    full_df = pd.read_csv('../data/TMDB-metadata-62K.csv')
    titles1 = pd.read_csv('../data/movies.csv')

    full_df['title_year'] = list(titles1.title)
    full_df['movieId'] = list(titles1.movieId)
    full_df = full_df[['movieId', 'title_year', 'overview', 'genres', 'vote_average', 'vote_count']]

    return [model1, full_df, titles1]


@st.cache(suppress_st_warning=True, show_spinner=False)
def make_dicts(titles):
    """
    Make dicts to transform ids to titles and doctags
    :param titles: make dictionaries to convert from movie id to doctags and titles
    :return: dicts
    """
    movieid_to_doctags = {movie: i for i, movie in enumerate(titles.movieId)}
    movieid_to_title = {i: title for i, title in zip(titles.movieId, titles.title)}

    return [movieid_to_doctags, movieid_to_title]


def two_random_movies(movies_df):
    """
    Select 2 pairs of movies randomly for playing
    :param movies_df: generates 2 pairs of random movies to play
    :return: 4 movies
    """
    n = len(movies_df)
    n1 = random.sample(range(n), 4)
    return movies_df.iloc[n1]


##################################

#Title
st.title('Ninjas versus Puppies')
st.markdown("**_A fun negotiation tool for movie selection_**")

# load model and data
model, data_df, titles = load_data()
movieid_to_doctags, movieid_to_title = make_dicts(titles)


# Sidebar title and page selector
st.sidebar.title('Ninjas versus Puppies')
st.sidebar.markdown('What do you want to do?')
pager = st.sidebar.radio('', ['Discover', 'Play', 'About'], index=0, key=None)

# Discover page
if pager == 'Discover':

    # Sidebar
    # st.sidebar.image(Image.open('../rawData/pictures/ninjas_versus_puppies.png'))
    st.markdown('Ninjas versus Puppies uses overviews, plot summaries and synopsis to discover relationship between movies.')
    st.sidebar.markdown('Discover allows you to find relationships between two movies...')
    st.sidebar.markdown('1) Type the movie you are looking for in the box. Parts of the title are ok!')
    st.sidebar.markdown('2) See if the movie appears in the selection menu below the box. If yes, select it!')
    st.sidebar.markdown('3) Repeat for the second movie')
    st.sidebar.markdown('4) Use the first slider to select the minimum rating that you would like for the results')
    st.sidebar.markdown('5) You can also increase or decrease the popularity of the movie (more votes, more popular)')
    st.sidebar.markdown('6) Finally, once you have your results, you can use the last slider to explore which movies are closer the the original ones by sliding to the extremes!')
    st.header('Your selection is:')

    # Movie 1 selection
    w1 = st.sidebar.text_input('Choose your first movie!', 'Dune (1984)', key=10)
    res1 = search_similar_title(w1, data_df.title_year, 50)
    try:
        ids1, word1_list = zip(*res1)
        title1 = st.sidebar.selectbox('Please select:', word1_list, key=11, index=22)
        movie1 = data_df[data_df['title_year'] == title1]['movieId']
    except:
        word1_list = []
        st.text('Please, tell me a bit more (maybe a year?)')
        title1 = st.sidebar.selectbox('Please select:', word1_list, key=11, index=22)

    # Movie2 selection
    w2 = st.sidebar.text_input('Choose your second movie!', "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
                               key=12)
    res2 = search_similar_title(w2, data_df.title_year, 50)
    try:
        ids2, word2_list = zip(*res2)
        title2 = st.sidebar.selectbox('Please select:', word2_list, key=13, index=5)
        movie2 = data_df[data_df['title_year'] == title2]['movieId']
    except:
        word2_list = []
        st.text('Please, tell me a bit more (maybe a year?)')
        title2 = st.sidebar.selectbox('Please select:', word2_list, key=13, index=5)

    # Put original images on the top
    try:
        target_image_list = put_target_images(movie1, movie2)
        st.image(target_image_list, width=200, caption = [movieid_to_title[movie1.iloc[0]],'', movieid_to_title[movie2.iloc[0]]])
        sim = movie_similarity(model, movieid_to_doctags, [movie1, movie2])
        print(movieid_to_title[movie1.iloc[0]], ' + ', movieid_to_title[movie2.iloc[0]],'Similarity:', sim)
    except:
        st.text('Choose two movies from the left...')

    min_average = st.sidebar.slider('Min Rating?', min_value=0.0, max_value=10.0, value=5.0, step=0.5, format=None,
                                    key=None)
    min_votes = st.sidebar.slider('Min Popularity?', min_value=0, max_value=10000, value=500, step=10, format=None,
                                  key=None)
    res_value = st.sidebar.slider('More like the first (0) or the second movie (6)?', min_value=0, max_value=6, value=3,
                                  step=1, format=None, key=None)

    # Create a copy of the data
    data = data_df.copy()

    # Filter by the slider values
    popular_movies = movie_filter(data, min_votes, min_average)

    # Create the slider movies and images
    try:
        res = make_slider_with_filter(model, popular_movies, movieid_to_doctags, movie1, movie2, 2000)
    except:
        res = []
        if len(word1_list) > 0 and len(word2_list) > 0:
            st.text('Maybe you need to reduce the minimum vote or the rating...')

    # If something found...
    if len(res) > 0:
        image = Image.open('../rawData/pictures/bracket.png')
        st.image(image, width=620)
        image_list = make_images(res, res_value)
        caption_list = [movieid_to_title[i] for i in res[res_value]]
        st.image(image_list, width=200, caption = caption_list)

        # Overview of the movies
        over = st.sidebar.checkbox('Want to see the overviews?', value=False, key=None)

        # Show the overviews if selected
        if over:
            st.table(data[data['movieId'].isin(res[res_value])][['title_year', 'overview', 'genres']])

        # Radio button to search similar movies to the found ones
        movies_list = [movieid_to_title[i] for i in res[res_value]]
        radio_selection = st.radio('Interested in similar movies to this one?', movies_list, index=0)
        movie_to_display = res[res_value][movies_list.index(radio_selection)]
        show_similar = st.button('Yes!', key=69)
        if show_similar:
            res2 = find_middle_with_filter(model, popular_movies, movieid_to_doctags,
                                           [movie_to_display, movie_to_display], 10,
                                           [movie_to_display, movie_to_display])

            if len(res2) > 0:
                # Make the Images!
                image_list2 = make_images(res2, None)
                st.image(image_list2, width=200)


# Play mode
elif pager == 'Play':

    #title and sidebar selectors
    st.sidebar.markdown('Can you guess relationships between movies?')
    st.sidebar.markdown("Select a minimum rating and popularity for your movies and try to guess between two pairs before the times end!")
    min_average2 = st.sidebar.slider('Min Rating?', min_value=0.0, max_value=10.0, value=5.0, step=0.5, format=None,
                                     key=None)
    min_votes2 = st.sidebar.slider('Min Popularity?', min_value=0, max_value=10000, value=500, step=10, format=None,
                                   key=None)

    # Create a copy of the data
    data2 = data_df.copy()

    # Filter data
    popular_movies_play = movie_filter(data2, min_votes2, min_average2)

    # Launch the guessing game
    play_button = st.sidebar.button('Play!')
    if play_button:
        st.subheader('Guess which two movies are related to these ones:')

        # select two pairs of movies randomly
        movies_play = two_random_movies(popular_movies_play)
        pairs = [(i, j) for i, j in zip(list(movies_play.movieId)[0::2], list(movies_play.movieId)[1::2])]

        # Select one pair
        correct_ones = pairs[0]

        # Create the images for the middle of the selected pair and make the images
        res3 = find_middle_with_filter(model, popular_movies_play, movieid_to_doctags, correct_ones, 2000, correct_ones)
        image_list = make_images(res3, None)
        caption_list = [movieid_to_title[i] for i in res3]
        st.image(image_list, width=200, caption=caption_list)

        # Randomized the pairs and show the options
        random.shuffle(pairs)
        for i, pair in enumerate(pairs):
            target_image_list = put_target_images([pair[0]], [pair[1]])
            st.header(['A', 'B'][i])
            st.image(target_image_list, width = 200, caption = [ movieid_to_title[pair[0]], '', movieid_to_title[pair[1]]] )

        # Makes a progress bar as timing. When ends show the result
        progress_bar = st.progress(0)
        for i in range(100):
            # Update progress bar.
            progress_bar.progress(i)
            time.sleep(0.2)
        st.header('The solution is ' + 'AB'[pairs.index(correct_ones)])

# About page
elif pager == 'About':

    # Title and plot button
    st.sidebar.markdown('Ninja versus Puppies is powered by DOC2VEC')
    st.markdown('More than 62K movies overviews, plots and synopsis have been converted to vectors '
            'using the DOC2VEC algorithm. Similarity between movies is computed by measuring the cosine'
            ' similarity of their vectors. In addition, the BERT embedding of the overviews was used to build '
            'the representation below. See if you can find your favourite movies!')
    plot_it = st.sidebar.button('Plot!')

    # plot the embeddings
    if plot_it:
        plotting_movies = plot_bokeh()
        st.bokeh_chart(plotting_movies)
        #plot_bokeh(m1=[], m2=[], res=[], res_value=[])
else:
    pass
