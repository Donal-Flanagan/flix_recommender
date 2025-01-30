import pandas as pd

def create_user_item_matrix(ratings, fill_value=None):
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


class DataTransformer():
    
    def __init__(self,
                data_path="../data/ml-latest-small/"):

        self.data_path = data_path
        
        self._ratings_df = pd.DataFrame()
        self._movies_df = pd.DataFrame()
        self._links_df = pd.DataFrame()
        self._genres_encoded_df = pd.DataFrame()
        self._genres_list = []
        
        self._load_movie_lens_data()
        self._create_binary_encoded_genres()
        self._create_genres_list()
        
    def _load_movie_lens_data(self):
        self._ratings_df = pd.read_csv(
            self.data_path+'ratings.csv',
            usecols=['userId', 'movieId', 'rating', 'timestamp'],
            dtype={'userId': int, 'movieId': int, 'rating': float},
            parse_dates=['timestamp'],  
            converters={'timestamp': lambda x: pd.to_datetime(int(x), unit='s')}
        )
        
        self._movies_df = pd.read_csv(
            self.data_path+'movies.csv',
            usecols=['movieId', 'title', 'genres'],
            dtype={'movieId': int, 'title': str, 'genres': str}
        )
        # Move the year values to a separate col
        self._movies_df["year"] = self._movies_df["title"].str.extract(r"\((\d{4})\)")
        self._movies_df["title"] = self._movies_df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
        self._movies_df["year"] = self._movies_df["year"].astype(float).astype("Int64")
        
        self._links_df = pd.read_csv(
            self.data_path+'links.csv', 
            usecols=['movieId', 'imdbId', 'tmdbId'],
            dtype={'movieId': int, 'imdbId': str, 'tmdbId': str}
        )

    def _create_binary_encoded_genres(self):
        self._genres_encoded_df = self._movies_df['genres'].str.get_dummies(sep='|')
        self._genres_encoded_df = pd.concat([self._movies_df[['movieId']], self._genres_encoded_df], axis=1)
        self._movies_df.drop('genres', axis=1, inplace=True)

    def _create_genres_list(self):
        self._genres_list = [genre for genre in self._genres_encoded_df.columns.values if genre != 'movieId']

    def get_ratings_df(self):
        return self._ratings_df.copy()
        
    def get_movies_df(self):
        return self._movies_df.copy()
        
    def get_links_df(self):
        return self._links_df.copy()

    def get_binary_encoded_genres_df(self):
        return self._genres_encoded_df.copy()

    def get_genres_list(self):
        return self._genres_list.copy()

    def get_genre_movies(self, genre):
        if genre not in self._genres_list:
            print(f"Please specify a valid genre from the following list: {', '.join(self._genres_list)}")
            return None
        return self._movies_df[self._movies_df.movieId.isin(self._genres_encoded_df[self._genres_encoded_df[genre]==1].movieId)].copy()
