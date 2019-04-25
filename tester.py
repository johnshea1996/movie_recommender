import pandas as pd
import numpy as np

pd.set_option("display.max_columns",4)

# Read movie data into a dataframe
movie_data_col = ['movie_id','movie_title']
movie_data = pd.read_csv("./ml-100k/u.item", sep='|', encoding='latin-1', names=movie_data_col, header=None, usecols=['movie_id','movie_title'])
# Read in the ratings data into a dataframe
ratings_data_col = ['user_id', 'movie_id', 'rating']
ratings_data = pd.read_csv("./ml-100k/u.data", sep="\t", names=ratings_data_col, header=None, usecols=['user_id', 'movie_id', 'rating'])
# Add movie titles to rating data
ratings_data = pd.merge(movie_data,ratings_data,on='movie_id')

group_movies = ratings_data.groupby('movie_id')
ratings_data = group_movies.filter(lambda x: x['rating'].count() > 100)

# Create matrix of all user ratings for every movie
movie_ratings_data = ratings_data.pivot_table(index='user_id',columns='movie_id',values='rating')
movie_ratings_data.fillna(0,inplace=True)

def get_user_array(udata):
    u = np.array(udata)
    return u

def cosine_similarity(u1, u2):
   return (np.dot(u1,u2))/(np.linalg.norm(u1)*np.linalg.norm(u2))

def get_predicted_rating(tuser,tmid):
    sims = []
    u1 = get_user_array(movie_ratings_data.loc[tuser,:])
    for user in movie_ratings_data.index:
        if(user != tuser):
            ui_rating = movie_ratings_data.loc[user,tmid]
            if(ui_rating > 0):
                ui = get_user_array(movie_ratings_data.loc[user,:])
                sims.append([user,cosine_similarity(u1,ui),ui_rating])
    rating = 0
    nrating = 0
    for i in range(len(sims)):
        rating += sims[i][1]*sims[i][2]
        nrating += abs(sims[i][1])
    return rating/nrating

def rec_movies(tu,num_rec):
    results = []
    for mid in movie_ratings_data.columns.values:
        pr = get_predicted_rating(tu,mid)
        tr = movie_ratings_data.loc[tu,mid]
        results.append([mid,pr,tr])
        print(mid)

    results.sort(key=lambda x: x[1],reverse=True)

    count = 0
    for i in range(len(results)):
        if(results[i][2] > 0):
            print(movie_data.loc[results[i][0],'movie_title'],results[i][1],results[i][2])
            count+=1
        if(count >= num_rec):
            break

#print(ratings_data.groupby('user_id')['rating'].count().sort_values(ascending=False))
rec_movies(13,10)

