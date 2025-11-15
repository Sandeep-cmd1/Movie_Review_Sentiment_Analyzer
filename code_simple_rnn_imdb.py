#SIMPLE RNN MODEL - USING IMDB DB of KERAS

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

## Load IMDB dataset
max_features = 10000 #Initialize vocab size
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=max_features) #imdb data is stored in OHE index values for each word
#num_words restrict total vocab to 10K most freq words, remove other words

#Print shape of the data
print(f'Training data shape: {X_train.shape}, Training labels shape:{y_train.shape}')
print(f'Testing data shape: {X_test.shape}, Training labels shape:{y_test.shape}')


#Inspect sample review and its label
len(X_train[0]) #Is a vector of indices of 1 value for a word in a review in a vector with dimension of max_features value --> One Hot Encoded

y_train[1]

#Inspect a sample review and its label
sample_review = X_train[0]
sample_label = y_train[0]

print(f'Sample review (as integers):{sample_review}')
print(f'Sample label (as integers):{sample_label}')

#Mapping of words index back to words (for our understanding)
word_index = imdb.get_word_index()
#word_index -> displays all words in {'word':index value} format, where index shows where it is 1 in vector, rest are 0s
reverse_word_index = {value:key for key, value in word_index.items()}
#reverse_word_index -> displays all words in {'index value':word} format, this is helpful to easily dislpay a word by calling its index (key)
reverse_word_index

#Decode OHE reviews in sample_review into words/ sentences
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review]) #indices added to words in this keras DB before adding reserved start tokens, so use index minus 3 always to call a word
decoded_review

#Padding reviews to make all review vectors of same size

max_len = 500
X_train = sequence.pad_sequences(X_train,maxlen=max_len) #deafult is pre-padding
X_test = sequence.pad_sequences(X_test,maxlen=max_len)
X_train[0]
len(X_train[0])

#Train Simple RNN

model = Sequential()
model.add(Embedding(max_features,128,input_length=max_len)) #Embedding layers
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

#Create an instance of an EarlyStopping callback

from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
earlystopping

#Train the model with earlystopping
history = model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2,callbacks=[earlystopping])


#Check trained model weights by
model.get_weights() 

#Save model as .h5 file
model.save('simple_RNN_imdb_r0.h5')

#Predict data by loading pre-trained model
model_predict = load_model('simple_RNN_imdb_r0.h5')
model_predict.summary()

#Helper Functions

#Function to encode a review of words into vectors and pad the vectors
def processing_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2)+3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

#Function to decode reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

#Prediction function
def predict_sentiment(review):
  encoded_review = processing_text(review)
  prediction = model.predict(encoded_review)
  sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
  return sentiment, prediction[0][0]

example_review = "The movie was fantastic! The acting was great and the plot was thrilling."

sentiment,prediction_score = predict_sentiment(example_review)
print(f'Review: {example_review}')
print(f'Sentiment: {sentiment}')
print(f'Prediction Score: {prediction_score}')




