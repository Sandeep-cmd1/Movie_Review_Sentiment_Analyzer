#Import libraries and load model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


#Load IMDB dataset word indedx to be used to encode/decode user review input
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

#Load pre-trained model with ReLU activation fn
model = tf.saved_model.load("my_simple_rnn_imdb_savedmodel")


#Helper functions for encoding & padding user reviews and decoding embedded reviews 
#Function to encode a review of words into vectors and pad the vectors
def processing_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2)+3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review
#Function to decode reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#Create prediction function
def predict_sentiment(review):
  encoded_review = processing_text(review)
  prediction = model.predict(encoded_review)
  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
  return sentiment, prediction[0][0]

#STREAMLIT APP DESIGN
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify as positive or negative.')

user_input = st.text_area('Enter a Movie Review')

if st.button('Classify'):
  encoded_padded_input = processing_text(user_input)

  prediction = model.predict(encoded_padded_input)
  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]}')

else:

  st.write('Please enter a movie review')





