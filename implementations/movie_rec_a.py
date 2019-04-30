import pandas as pd
import numpy as np
import random
import datetime

pd.set_option("display.max_columns",4)

MIN_RATINGS = 150
# Read movie data into a dataframe
movie_data = pd.read_csv("../ml-100k/u.item", sep='|', encoding='latin-1',names=['movie_id', 'movie_title'], header=None, usecols=['movie_id','movie_title'])
# Read in the ratings data into a dataframe
ratings_data = pd.read_csv("../ml-100k/u.data", sep="\t", names = ['user_id', 'movie_id', 'rating'], header=None, usecols=['user_id', 'movie_id', 'rating'])
user_movie_ratings = ratings_data.pivot_table(index='user_id',columns='movie_id',values='rating',fill_value=0)


def add_rating(rating_data, mid, new_rating):
    return rating_data.append({'user_id':0,'movie_id':mid,'rating':new_rating},ignore_index=True)

def get_user_array(udata):
    u = np.array(udata)
    return u

def cosine_similarity(u1, u2):
   return (np.dot(u1,u2))/(np.linalg.norm(u1)*np.linalg.norm(u2))

# Gets predicted rating for a target user and target movie
def get_predicted_rating(tuser,tmid,movie_ratings_data):
    sims = []
    u1 = get_user_array(movie_ratings_data.loc[tuser,:])
    for user in movie_ratings_data.index:
        # Do not compare for the same user
        if(user != tuser):
            ui_rating = movie_ratings_data.loc[user,tmid]
            if(ui_rating > 0):
                ui = get_user_array(movie_ratings_data.loc[user,:])
                sims.append([user,cosine_similarity(u1,ui),ui_rating])
    rating = 0
    sim_sum = 0
    # Calculate rating as a weighted sum of other user's ratings and cosine similarity
    for i in range(len(sims)):
        rating += sims[i][1]*sims[i][2]
        sim_sum += abs(sims[i][1])
    return rating/sim_sum


def rec_movies(user_ratings):
    # Add new user ratings to the new ratings data
    new_ratings_data = ratings_data
    for mid, rating in user_ratings.items():
        new_ratings_data = add_rating(new_ratings_data, mid, rating)
    # Remove any movies that do not meet the minimum rating requirement
    group_movies = new_ratings_data.groupby('movie_id')
    new_ratings_data = group_movies.filter(lambda x: x['rating'].count() > MIN_RATINGS)
    # Create user to movie rating table
    movie_ratings_data = new_ratings_data.pivot_table(index='user_id', columns='movie_id', values='rating',fill_value=0)

    results = []
    x = 0
    for mid in movie_ratings_data.columns.values:
        pr = get_predicted_rating(0,mid,movie_ratings_data)
        results.append([mid,pr])
        print("\rRecommending Movies ... {}%".format(int(x/len(movie_ratings_data.columns.values)*100)),end="")
        x += 1

    results = [i for i in results if i[0] not in user_ratings.keys()]
    results.sort(key=lambda x: x[1],reverse=True)
    return results

def test(tu,out_file):
    # Base ratings
    base_ratings_df = pd.read_csv("../ml-100k/ua.base", sep="\t", names=['user_id', 'movie_id', 'rating'], header=None, usecols=['user_id', 'movie_id', 'rating'])
    user_movie_base_ratings = base_ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating')

    # Test ratings
    test_ratings_df = pd.read_csv("../ml-100k/ua.test", sep="\t", names=['user_id', 'movie_id', 'rating'], header=None, usecols=['user_id', 'movie_id', 'rating'])
    user_ratings_df = test_ratings_df[test_ratings_df['user_id'] == tu]
    user_ratings = user_ratings_df.pivot_table(columns='movie_id', index='user_id', values='rating').to_dict('list')
    for mid, data in user_ratings.items():
        user_ratings[mid] = data[0]

    avg_rating = base_ratings_df[base_ratings_df['user_id']==tu]['rating'].mean()
    movies_seen = base_ratings_df[base_ratings_df['user_id'] == tu]['rating'].count()
    results = rec_movies(user_ratings)

    print("\nResults")
    out_file.write("\nResults\n")
    count = 0
    above_avg= 0
    for mid, pr in results:
        if (user_movie_ratings.loc[tu, mid] > 0):
            title = movie_data.loc[mid, 'movie_title']
            true_rating = user_movie_ratings.loc[tu, mid]
            print(title, true_rating)
            if(true_rating >= avg_rating):
                above_avg += 1
            count += 1
        if (count >= movies_seen*0.2  or count>=10):
            break
    print("Seen {} movies".format(movies_seen))
    print("Average Rating {}".format(avg_rating))
    print("{}/{} movies were above the average".format(above_avg,count))
    out_file.write("Seen {} movies\n".format(movies_seen))
    out_file.write("Average Rating {}\n".format(avg_rating))
    out_file.write("{}/{} movies were above the average\n".format(above_avg,count))
    return above_avg/count

test_users = [random.randint(0,943) for i in range(10)]
results = []
out_file = open("../test_results/movie_rec_a/test12_min150.txt",'w')

start_time = datetime.datetime.now()
for i in range(len(test_users)):
    print("\nTest {} - Running for User {}".format(i,test_users[i]))
    out_file.write("\nTest {} - Running for User {}".format(i,test_users[i]))
    results.append(test(test_users[i],out_file))
end_time = datetime.datetime.now()

print("\nAverage result {}".format(np.mean(results)))
out_file.write("\nAverage result {}\n".format(np.mean(results)))

elapsed_time = end_time-start_time
print("Elapsed time - {}.{}s".format(elapsed_time.seconds,elapsed_time.microseconds))
out_file.write("Elapsed time {}.{}".format(elapsed_time.seconds,elapsed_time.microseconds))