import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spotipy
from spotipy.oauth2 import SpotifyOAuth

CLIENT_ID = '42b4676e86474f108fddcdc92c5bf787'
CLIENT_SECRET = '7939744af53c40e28a93514318dbc3a7'
REDIRECT_URI = 'http://localhost:8888/callback'
SCOPE = "user-library-read user-top-read playlist-modify-public"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope=SCOPE))

songs_data = pd.read_csv(r'songs_data.csv')
song_vectorizer = CountVectorizer()
song_vectors = song_vectorizer.fit_transform(songs_data['Genre'])
vectorizer = TfidfVectorizer(stop_words='english')
song_vectors = vectorizer.fit_transform(songs_data['Song-Name'])

def get_similarities(song_name, data):
    song_vector = vectorizer.transform([song_name])
    similarities = cosine_similarity(song_vector, song_vectors)
    return similarities.flatten()

def recommend_songs(song_name, data=songs_data):
    songs_data = data[data['Song-Name'].str.lower() == song_name.lower()]
    artist_name = songs_data['Singer/Artists'].values[0]
    
    if songs_data.shape[0] == 0:
        st.write('This song is either not so popular or you have entered an invalid name.\nSome songs you may like:\n')
        recommend_songs = data.sample(n=5)
        for song in recommend_songs['Song-Name'].values:
            st.write(song)
        return

    st.markdown(f"### Your Song: {song_name} by {artist_name}")
    
    # Search for the song on Spotify
    result = sp.search(q=f"track: {song_name} artist: {artist_name}", type='track', limit=1)
    if result['tracks']['items']:
        track = result['tracks']['items'][0]
        track_name = track['name']
        track_url = track['external_urls']['spotify']
        track_album = track['album']['name']
        track_thumbnail = track['album']['images'][0]['url']
        
        st.markdown(f"**Now Playing**: {track_name}")
        st.write(f"Album: {track_album}")
        st.write(f"Listen on Spotify: [Link]({track_url})")
        st.image(track_thumbnail, use_column_width=True)

    st.write("\nRecommended songs:")
    similarities = get_similarities(song_name, data)
    
    if similarities is None or similarities.shape[0] == 0:
        st.write(f"Error: No similarities found for {song_name}. Please check the feature vectorization.")
        return

    data['similarity_factor'] = similarities
    data.sort_values(by=['similarity_factor', 'User-Rating'], ascending=[False, False], inplace=True)

    recommended_songs = data[['Song-Name', 'Singer/Artists']].iloc[1:11]  

    rows = 2  
    for start_idx in range(0, len(recommended_songs), rows):  
        end_idx = min(start_idx + rows, len(recommended_songs))
        cols = st.columns(rows)
        
        for idx, col in enumerate(cols):
            if start_idx + idx < len(recommended_songs):
                song = recommended_songs.iloc[start_idx + idx]
                song_name = song['Song-Name']
                artist_name = song['Singer/Artists']
                result = sp.search(q=f"track: {song_name} artist: {artist_name}", type='track', limit=1)
                
                if result['tracks']['items']:
                    track = result['tracks']['items'][0]
                    track_name = track['name']
                    track_url = track['external_urls']['spotify']
                    track_album = track['album']['name']
                    track_thumbnail = track['album']['images'][0]['url']
                    
                    with col:
                        st.subheader(f"Track: {track_name}")
                        st.write(f"Album: {track_album}")
                        st.write(f"Listen on Spotify: [Link]({track_url})")
                        st.image(track_thumbnail, use_column_width=True)

st.title('Song Recommendation System')
song_name = st.text_input("Enter a song name:")

if song_name:
    recommend_songs(song_name)
