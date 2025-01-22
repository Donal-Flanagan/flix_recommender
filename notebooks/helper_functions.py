import pandas as pd

def create_ratings_df(ratings, fill_value=None):
    """
    Pivot the user-movie ratings into a DataFrame, also known as a user-item matrix.

    Args:
        ratings (pd.DataFrame): A DataFrame containing at least:
            - 'userId': User identifier.
            - 'movieId': Movie identifier.
            - 'rating': Ratings given by users to movies.
        fill_value: A value with which to fill NaN values (missing/sparse data)

    Returns:
        pd.DataFrame: A pivot table with:
            - Index: userId (each row represents a user).
            - Columns: movieId (each column represents a movie).
            - Values: Ratings given by users to movies. Range: 0.5 - 5 (NaN where no rating exists).
    """
    return pd.pivot_table(
        data=ratings,
        values='rating',
        index='userId', 
        columns='movieId',
        fill_value=fill_value     
    )

def get_movie_id_from_title(movie_title, movies):
    """
    Retrieve the movie ID corresponding to a given movie title.

    Args:
        movie_title (str): The title of the movie to search for. The search is case-insensitive.
        movies (pd.DataFrame): A DataFrame containing movie metadata, with at least the following columns:
            - 'movieId': The unique identifier for each movie (int).
            - 'title': The title of the movie (str).

    Returns:
        int: The movie ID corresponding to the given movie title.

    Raises:
    ------
    ValueError
        If the movie title is not found in the DataFrame or more than one movie with that title are found in the DataFrame.
    """
    movie_title_mask = movies['title'].str.contains(movie_title, case=False)
    matching_movie_ids = movies.loc[movie_title_mask, 'movieId']

    # Handle cases where no movies match
    if matching_movie_ids.empty:
        raise ValueError(f"No movie found with title matching '{movie_title}'. Please check the title and try again.")

    # Handle cases where multiple movies match
    if len(matching_movie_ids) > 1:
        raise ValueError(
            f"Multiple movies found with title matching '{movie_title}'. Please provide more details.\n"
            f"Matching movie IDs: {matching_movie_ids.values.tolist()}"
        )
        
    return matching_movie_ids.values[0]

def get_top_ten_similar_movies(similar_movies, movies, movie_id):
    """
    Retrieves the top 10 most similar movies to a given movie and adds the movie titles.

    Args:
        similar_movies (pd.DataFrame): A DataFrame where:
            - The index is 'movieId', containing the IDs of movies being compared.
            - A single column entitled with the movie ID of the target movie.
            - The values in the column are similarity scores between the target movie
              and other movies.
        movies (pd.DataFrame): A DataFrame containing movie metadata, with at least the following columns:
            - 'movieId': The unique identifier for each movie (int).
            - 'title': The title of the movie (str).
        movie_id (int): The movie ID for which similar movies are being retrieved. 

    Returns:
        pandas.DataFrame: A DataFrame containing the top 10 similar movies, with the following columns:
            - Similarity scores (from the `similar_movies` DataFrame).
            - 'title': The title of each similar movie.
            - The 'movieId' serves as the index of the DataFrame.
    """
    ranked_by_similarity = similar_movies.sort_values(by=movie_id, ascending=False)
    top_ten = ranked_by_similarity.head(10).copy()
    top_ten.reset_index(inplace=True)
    # Add the titles
    top_ten = top_ten.merge(movies[['movieId', 'title']], on='movieId', how='left')
    top_ten.set_index('movieId', inplace=True)

    return top_ten
