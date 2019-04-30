import matplotlib.pyplot as plt
import pandas as pd

ratings_data = pd.read_csv("../ml-100k/u.data", sep="\t", names = ['user_id', 'movie_id', 'rating'], header=None, usecols=['user_id', 'movie_id', 'rating'])
uids = ratings_data.groupby(by='user_id')['rating'].count().index
uvals= ratings_data.groupby(by='user_id')['rating'].count().values
mids = ratings_data.groupby(by='movie_id')['rating'].count().index
mvals =ratings_data.groupby(by='movie_id')['rating'].count().values

f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.hist(uvals,bins=50)
ax1.set_title("Number of Ratings by User")
ax1.set(xlabel="Number of Ratings",ylabel="Frequency")
ax2.hist(mvals,bins=50)
ax2.set_title("Number of Ratings by Movie")
ax2.set(xlabel="Number of Ratings",ylabel="Frequency")
plt.show()