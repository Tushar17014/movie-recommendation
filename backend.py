import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def findCosineSimilarity(l):
    sim = []

    for i in l:
        temp = []
        for j in l:
            dot = np.dot(i, j)
            norm1 = np.linalg.norm(i)
            norm2 = np.linalg.norm(j)
            cosine_similar = dot / (norm1 * norm2)
            temp.append(cosine_similar)
        sim.append(temp)
    return sim

def convertTitle(t):
    s = ""
    for i in t:
        if i.isalnum() :
            s += i
        if i == ' ':
            s += i
    s = s.lower()
    return s

movies = pickle.load(open('movie_data.pkl', 'rb'))
Movies = pd.DataFrame(movies)
movies = pd.DataFrame(movies)
movies['title'] = movies['title'].apply(convertTitle)

cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()

l = []

for i in vector:
    l.append(np.array(i))

similarity = findCosineSimilarity(l)


def recommend(movieName):
    movieIndex = movies[movies['title'] == movieName].index[0]
    distances = sorted(list(enumerate(similarity[movieIndex])),reverse=True,key = lambda x: x[1])[1:6]
    for i in distances:
        print(Movies.iloc[i[0]].title)

