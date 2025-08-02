# 🎬 IMDB Movie Review Sentiment Classifier

An end-to-end NLP pipeline that classifies movie reviews as **positive** or **negative** using a **Recurrent Neural Network (RNN)**. This project covers everything from data handling and model development to deployment via a simple Gradio-powered web app.

## 🚀 Key Highlights

- 📂 Loads IMDB dataset directly from a ZIP file  
- 🧼 Cleans and tokenizes text with padding to fixed length  
- 🧠 Uses TensorFlow/Keras with an Embedding + Simple RNN architecture  
- 🧪 Splits dataset into train and test sets  
- 📈 Monitors accuracy and loss during training  
- 💾 Saves both the trained model (`model.h5`) and tokenizer (`tokenizer.pkl`)  
- 📣 Predicts sentiment from user-entered reviews  
- 🌍 Deployed via **Gradio** for real-time inference

## 📚 Dataset Details

- **Source**: [Kaggle - IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- **Columns**:  
  - `review`: textual movie review  
  - `sentiment`: label (either “positive” or “negative”)

## ⚙️ How It Works

1. **Data Loading**: The dataset is extracted from a zip archive and loaded into a pandas DataFrame.
2. **Label Encoding**: Sentiments are converted to binary format (positive → 1, negative → 0).
3. **Preprocessing**:  
   - Text is tokenized using Keras' `Tokenizer`  
   - Sequences are padded to a max length of **200** tokens
4. **Model Architecture**:  
   - **Embedding Layer** to represent words in vector form  
   - **Simple RNN** to capture temporal structure  
   - **Dense Output Layer** with sigmoid for binary classification
5. **Training**:  
   - Dataset split into 80% training and 20% testing  
   - Trained for 5 epochs with batch size 64
6. **Saving & Deployment**:  
   - Model and tokenizer are saved  
   - A **Gradio interface** is built for end-user predictions

## 📊 Example Output Screenshots

<img width="1772" height="348" alt="image" src="https://github.com/user-attachments/assets/48877df1-d674-4a19-82d0-8838d5cd1c3c" />
<img width="1795" height="379" alt="image" src="https://github.com/user-attachments/assets/a3313142-d672-42db-8e63-0f1abd1ae244" />



## 🌐 Live Demo

👉 [Click here to try it live](https://556a3208384d20d753.gradio.live/)

