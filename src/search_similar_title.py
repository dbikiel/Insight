import difflib


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
    res = difflib.get_close_matches(pattern, title_list, 5, threshold)
    best = []
    for i in res:
        if i in title_list:
            best.append((title_list.index(i), i))
    return best
