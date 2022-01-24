# Spotify Machine Learning
A cool way to predict the mood of music tracks with Neural Networks models using Keras and Tensorflow Libraries on Python. Music is a powerful language to express our feelings and in many cases is used as a therapy to deal with tough moments in our lives. Emotions and moods can be easily reflected in music, when we are doing sports, we tend to listen to energetic music, similarly when we are anxious or tired a nice relaxed song can help us to calm down. That’s why I try to figure out how classification models could help to determinate which is the mood of a specific track.
Required Tools:
- Pandas and Numpy for data analysis.
- Keras and Tensorflow to build the Deep Learning model.
- Sklearn to validate the model.
- Seaborn and Matplotlib to plot a nice graph.
- Spotipy Python Library (click here for more info).
- Spotify Credentials to access their Apis and Data acquisition (click here for more info).

### Spotify Audio Features:
Spotify uses a series of different features to classify tracks. [ Information from the Spotify Webpage. ]
- Acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- Danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- Energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- Instrumentalness: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- Liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides a strong likelihood that the track is live.
- Loudness: the overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing the relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.
- Speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- Valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- Tempo: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, the tempo is the speed or pace of a given piece and derives directly from the average beat duration.
### 1. Explaining and Analysing the Data:
To obtain the data I had to create a series of functions using the Spotipy Library. This library helps to automate the Spotify services downloading more technical information (explaining above) about playlists, songs, artist music, etc. For the main purpose of this article, I’ll not mention how I obtained the data, but I’ll explain what the data consists of.
As you may know, Classification problems use labeled data, so I had to create these labels. I decided to create 4 categories to label the tracks, these categories are “Energetic”, ”Calm”, “Happy” and “Sad”. I choose these categories based on the following article, who explains what is the best way of classifying music by mood.
![Link to Documentation 📄](https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/)
Then I searched on Spotify some playlists with different music tracks based on these 4 labels (200 tracks per label) and finally, I concatenated all these tracks into the main data frame labeled by each mood. The main data have 800 rows and 18 columns, but for information reduction purposes I decided to use the features of Length, Danceability, Acousticness, Energy, Instrumentalness, Liveness, Valence, Loudness, Speechiness and Tempo because they have more influence to classify the tracks. I grouped the data frame by labels calculating the mean of the tracks’ features. I obtained the following result:

![Data Frame grouped using mean stats.](https://raw.githubusercontent.com/heysouravv/Spotify-Machine-Learning/main/images/Screenshot%202022-01-24%20at%208.55.36%20AM.png)

Doing this simple analysis I quickly noticed that the most popular songs are Happy, Sad songs tend to have a long length, Energetic songs are most fast in tempo, and Calm songs tends to be acousticness.
