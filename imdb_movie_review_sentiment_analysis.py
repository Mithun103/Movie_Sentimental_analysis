import zipfile

zip_path = '/content/archive (5).zip'
extract_path = '/content/unzipped_folder'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/content/unzipped_folder/IMDB Dataset.csv")

data.head()

data.shape

type(data)

data.tail()

data["sentiment"].value_counts()

# positive -> 1
# negative -> 0
data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

data.head()

data.tail()

data["sentiment"].value_counts()

!pip install tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)

train_data.shape

test_data.shape

tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_data["review"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)

X_train

X_test

Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]

Y_train

Y_test

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))  # specify input_length
model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

# Now build the model
model.build(input_shape=(None, 200))  # (batch_size, sequence_length)

model.summary()

model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, Y_train, epochs = 5, batch_size = 64, validation_split = 0.2)

model.save("model.h5")

import joblib
joblib.dump(tokenizer, "tokenizer.pkl")

loss, accuracy = model.evaluate(X_test, Y_test)

print(loss)

print(accuracy)

def predictive_system(review):
  sequences = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequences, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

predictive_system("This movie was fantastic and amazing")

predictive_system("A trilling adventure with stunning visual")

predictive_system("A visual masterpiece")

predictive_system("Overall long and slow")

from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("/content/model.h5")
tokenizer = joblib.load("/content/tokenizer.pkl")

def predictive_system(review):
  sequences = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequences, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

review_sentiment = predictive_system("This movie was fantastic and amazing")

review_sentiment

!pip install gradio

import gradio as gr
title = "MOVIE SENTIMENT ANALYSIS APPLICATION"

app = gr.Interface(fn = predictive_system, inputs="textbox", outputs="textbox", title=title)

app.launch(share=True)

