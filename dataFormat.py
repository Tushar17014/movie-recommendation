import numpy as np
import pandas as pd
import ast
import pickle
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def stem(word):
    l = []
    for i in word.split():
        l.append(ps.stem(i))
    
    return " ".join(l)

def DicToList(d):
    l = []
    for i in ast.literal_eval(d):
        l.append(i['name'])
    return l

def DicToList3(d):
    l = []
    counter = 0
    for i in ast.literal_eval(d):
        if counter < 3:
            l.append(i['name'])
        counter+=1
    return l

def getDirector(d):
    l = []
    for i in ast.literal_eval(d):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l

def removeSpace(L):
    l = []
    for i in L:
        l.append(i.replace(" ",""))
    return l


def format(movies, credits):

    movies = movies.merge(credits, on="title")
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(DicToList)
    movies['keywords'] = movies['keywords'].apply(DicToList)
    movies['cast'] = movies['cast'].apply(DicToList3)
    movies['crew'] = movies['crew'].apply(getDirector)

    movies['genres'] = movies['genres'].apply(removeSpace)
    movies['keywords'] = movies['keywords'].apply(removeSpace)
    movies['cast'] = movies['cast'].apply(removeSpace)
    movies['crew'] = movies['crew'].apply(removeSpace)

    movies['overview'] = movies['overview'].apply(lambda x:x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
    new['tags'] = new['tags'].apply(lambda x: " ".join(x))
    new['tags'] = new['tags'].apply(lambda x: x.lower())
    new['tags'] = new['tags'].apply(stem)
    
    return new


movies = pd.read_csv('Data/tmdb_5000_movies.csv')
credits = pd.read_csv('Data/tmdb_5000_credits.csv')

pickle.dump(format(movies, credits).to_dict(), open('movie_data.pkl', 'wb'))

