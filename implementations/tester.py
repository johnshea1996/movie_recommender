import pandas as pd

ratings_data = pd.read_csv("../ml-100k/u.data", sep="\t", names = ['user_id', 'movie_id', 'rating'], header=None, usecols=['user_id', 'movie_id', 'rating'])
print(ratings_data[ratings_data['user_id']==196].sort_values(by='movie_id'))