import numpy as np
import pandas as pd
import random

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
    final = res.sort_values(by=['mean_ineq_sim'], ascending=False).head(5)
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

def movie_filter(movies_df, min_votes = 0, min_average = 0):
    return movies_df[(movies_df.vote_average >= int(min_average)) & (movies_df.vote_count >= int(min_votes))]
