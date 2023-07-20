import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the data from the csv file into a pandas dataframe
movies_data = pd.read_csv('movies.csv')

# Printing the first 5 rows of the dataframe to check the data
movies_data.head()

# Getting the number of rows and columns in the data frame
movies_data.shape

# Selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']

# Replacing the null values with an empty string for the selected features
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combining all the 5 selected features into a single string for each movie
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + \
                  movies_data['cast'] + ' ' + movies_data['director']

# Converting the combined text data into feature vectors using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculating the cosine similarity matrix of the feature vectors
similarity = cosine_similarity(feature_vectors)

# Getting the movie name from the user
movie_name = input('Enter your favorite movie name: ')

# Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()

# Finding the closest match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]

# Finding the index of the movie with the title that matches the closest to the user input
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# Getting a list of similar movies and their similarity scores based on the index
similarity_score = list(enumerate(similarity[index_of_the_movie]))

# Sorting the movies based on their similarity score in descending order
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# Printing the names of similar movies to the user
print('Movies suggested for you: \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i <= 30:  # Displaying up to 30 similar movies
        print(i, '.', title_from_index)
        i += 1
