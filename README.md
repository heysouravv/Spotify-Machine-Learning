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

- [Link to Documentation for Music Mood Classification 📄](https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/)

Then I searched on Spotify some playlists with different music tracks based on these 4 labels (200 tracks per label) and finally, I concatenated all these tracks into the main data frame labeled by each mood. The main data have 800 rows and 18 columns, but for information reduction purposes I decided to use the features of Length, Danceability, Acousticness, Energy, Instrumentalness, Liveness, Valence, Loudness, Speechiness and Tempo because they have more influence to classify the tracks. I grouped the data frame by labels calculating the mean of the tracks’ features. I obtained the following result:

![Data Frame grouped using mean stats.](https://raw.githubusercontent.com/heysouravv/Spotify-Machine-Learning/main/images/Screenshot%202022-01-24%20at%208.55.36%20AM.png)

Doing this simple analysis I quickly noticed that the most popular songs are Happy, Sad songs tend to have a long length, Energetic songs are most fast in tempo, and Calm songs tends to be acousticness.

### 2. Building the Model:
- 2.1- Pre-Processing the Data:
To normalize the features I used MinMaxScaler to scale the values between a range of [0,1] and preserving the shape of the original distribution. I also encoded the 4 labels because Neural Networks uses numerical values to train and test. Finally, I split the data by 80% for training and 20% for testing.
- 2.2 Creating the model:
To build the model I used the library Keras, this library is designed to enable fast experimentation with Deep Neural Networks, focused on being user-friendly. My main goal is to classify tracks in the 4 categories of moods (Calm, Energetic, Happy and Sad) so my model consists of a Multi-Class Neural Network with an input of 10 Features, 1 Layer with 8 nodes, and 4 outputs with the output Layer. I also need to use a Classifier as an Estimator, in this case, the Classifier is KerasClassifier, which takes as an argument a function that I created previously with the Neural Network model defined. The activation Function corresponds to a Rectified Linear Unit (Relu), the Loss function is a Logistic Function and Adam Gradient Descent Algorithm is the optimizer.
-- Important: I disabled the eager execution and v2 behavior of TensorFlow because I keep trying to understand and to learn how the library works in those modes
- 2.3 Evaluating the model:
Using K-Fold Cross Validation I evaluated the estimator using the train data. The number of splits is K=10 shuffling all the values. The Accuracy of the model is the average of the accuracy of each fold, in this case, the Accuracy was **72.75%**.
- 2.4 Training the Model:
It’s important to mention that the model was trained with **640 samples** (80% of the main data).
- 3. Accuracy of the Multi-Class Neural Network:
Finally to evaluate the accuracy of the model I plotted a Confusion Matrix using Seaborn Library and Matplotlib. I also calculated the accuracy score provided by Sklearn Library. With a Final Accuracy score of 76% and taking a look at the Confusion Matrix, I noticed my model is good classifying Calm and Sad songs, but it’s having some issues dealing with Energetic and Happy songs. I could modify some parameters like the batch size, epochs, or maybe aggregate or delete some track features to train my model and thus help to improve the accuracy of the model.
