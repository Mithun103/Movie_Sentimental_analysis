from keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model.h5")
tokenizer = joblib.load("tokenizer.pkl")

def predictive_system(review):
  sequences = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequences, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

review_sentiment = predictive_system("This movie was fantastic and amazing")

review_sentiment


import gradio as gr
title = "MOVIE SENTIMENT ANALYSIS APPLICATION"

app = gr.Interface(fn = predictive_system, inputs="textbox", outputs="textbox", title=title)

app.launch(share=True)
