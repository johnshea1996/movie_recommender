# movie_rec_c.py

from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import prediction_algorithms
import pandas as pd
import random
import datetime
import numpy as np

# Create pandas dataframes on the various data
ratings_data = pd.read_csv("../ml-100k/u.data", sep="\t", names = ['user_id', 'movie_id', 'rating'], header=None, usecols=['user_id', 'movie_id', 'rating'])
movie_data = pd.read_csv("../ml-100k/u.item", sep='|', encoding='latin-1',names=['movie_id', 'movie_title'], header=None, usecols=['movie_id','movie_title'])

# Create the Suprise dataset to be used for this system
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file("../ml-100k/u.data",reader=reader)
# Choose the SVD Matrix Factorization algorithm
algo = prediction_algorithms.KNNWithMeans()
# Run a 5-fold cross validation the algorithm ot get the RMSE
results = cross_validate(algo,data,measures=['rmse','mae'],cv=5, verbose=True)

# Create an 80/20 split of the data into a training and testing set
trainset,testset = train_test_split(data,test_size=0.20)

# Begin training the algorithm
start_time = datetime.datetime.now()
algo.fit(trainset)

# Get predictions using the testing set
predictions = algo.test(testset)

# Test takes in a random user, predictions from the test set, and a output file to write to
# Finds the given predictions for that user from the test set, finds the top predictions in order,
# and tries to calculate how many of those top movies whose predicted ratings are higher than the
# the user's average rating have a real value that is also higher than the user's average rating
def test_user_results(tu,predictions,out_file):
    # Filter out to target user's predictions
    user_predictions = [i for i in predictions if i[0] == str(tu)]
    user_predictions.sort(key=lambda x: float(x[3]),reverse=True)

    avg_rating = ratings_data[ratings_data['user_id']==tu]['rating'].mean()
    movies_seen = ratings_data[ratings_data['user_id']==tu]['rating'].count()

    count = 0
    above_avg = 0
    for prediction in user_predictions:
        pred_rating = prediction[3]
        if(pred_rating>avg_rating):
            title = movie_data.loc[int(prediction[1]),'movie_title']
            true_rating = prediction[2]
            print(title, true_rating, pred_rating)
            if true_rating >= (avg_rating-0.10):
                above_avg += 1
            count += 1
            if count>=0.2*movies_seen or count>=10:
                break
    print("Seen {} movies".format(movies_seen))
    print("Average Rating {}".format(avg_rating))
    print("{}/{} movies were above the average".format(above_avg,count))
    out_file.write("Seen {} movies\n".format(movies_seen))
    out_file.write("Average Rating {}\n".format(avg_rating))
    out_file.write("{}/{} movies were above the average\n".format(above_avg, count))
    return above_avg/count


results = []
out_file = open("../test_results/movie_rec_c/test10.txt",'w')

# Test the results of the predictions on the test set for 10 randomly selected users
# who get at least one recommended movie from test set
while len(results) < 10:
    tu = random.randint(0,943)
    if len([p for p in predictions if p[0] == str(tu)]) > 5:
        print("\nTest {} - Running for User {}".format(len(results), tu))
        out_file.write("\nTest {} - Running for User {}\n".format(len(results), tu))
        results.append(test_user_results(tu,predictions,out_file))
end_time = datetime.datetime.now()

# Remove any -1 from results, indicating that no movies were recommended in the first place

print("\nAverage result {}".format(np.mean(results)))
out_file.write("\nAverage result {}\n".format(np.mean(results)))

elapsed_time = end_time - start_time
print("Elapsed time - {}.{}s".format(elapsed_time.seconds, elapsed_time.microseconds))
out_file.write("Elapsed time {}.{}".format(elapsed_time.seconds, elapsed_time.microseconds))

