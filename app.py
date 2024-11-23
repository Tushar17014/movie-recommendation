import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movies = pickle.load(open('movie_data.pkl', 'rb'))
movies = pd.DataFrame(movies)

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vector)


def getPoster(m_id):
    url = "https://api.themoviedb.org/3/movie/{}?language=en-US".format(m_id)

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1Njc0OGE4NDdjNzljZTdhOTNmYzY2ZWFiNTlhNWE5MCIsInN1YiI6IjY1Njc2YzQxYTM0OTExMDExYjU5YmRlMSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ._n1KzdMx64UlqvAwKfM0_NVvT6_uImvSg_XlkhFQmL0"
    }

    response = requests.get(url, headers=headers)
    data = response.json()
    return "https://image.tmdb.org/t/p/original" + data['poster_path']


def recommend(movieName):
    movieIndex = movies[movies['title'] == movieName].index[0]
    distances = sorted(
        list(enumerate(similarity[movieIndex])), reverse=True, key=lambda x: x[1])[1:6]
    rec_movies = []
    rec_movies_posters = []
    for i in distances:
        mid = movies.iloc[i[0]].movie_id
        rec_movies.append(movies.iloc[i[0]].title)
        rec_movies_posters.append(getPoster(mid))
    return rec_movies, rec_movies_posters


st.title("Movie Recommendation")
movieName = st.selectbox(
    'How would you like to be contacted?', movies['title'].values)

if st.button("Search", type="primary"):
    rec, poster = recommend(movieName)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(rec[0])
        st.image(poster[0])
    with col2:
        st.text(rec[1])
        st.image(poster[1])
    with col3:
        st.text(rec[2])
        st.image(poster[2])
    with col4:
        st.text(rec[3])
        st.image(poster[3])
    with col5:
        st.text(rec[4])
        st.image(poster[4])


    # for i in rec:
    #     st.write(i)
