import streamlit as st
import dill
import pickle
import nltk
import re
import csv
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from subprocess import run

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = dill.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab') 
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')  

    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    normalized_tokens = [token.lower() for token in filtered_tokens]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in normalized_tokens]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]

    return lemmatized_tokens

# Classification function
def classify_text(text):
    probability = model.predict_proba([text])[0][1]  # Probability of spam
    threshold = 0.7
    return "spam" if probability > threshold else "ham"

# Feedback file path
feedback_file = 'feedback.csv'

# Streamlit app
st.title("Spam Classifier")
st.write("This is an application to classify emails as spam or ham.")

# Input fields
message = st.text_area("Enter a message to classify:")
classify_button = st.button("Classify")

# Classification logic
if classify_button:
    if not message.strip():
        st.error("Please enter a valid message before classifying.")
    else:
        processed_message = preprocess_text(message)
        vectorized_message = vectorizer.transform([processed_message])
        prediction = model.predict(vectorized_message)[0]
        classification = "ham" if prediction == 1 else "spam"

        if prediction == 1:
            st.success(f"The message is classified as **{classification}**.")
        else:
            st.error(f"The message is classified as **{classification}**.")
        
        # Feedback section
        st.markdown("---")
        st.subheader("Feedback")
        st.write("Your feedback helps us improve our model!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Correct"):
                with open(feedback_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([message, classification, 1])  # 1 for correct
                st.success("Thank you for your feedback!")

        with col2:
            if st.button("Incorrect"):
                with open(feedback_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([message, classification, 0])  # 0 for incorrect
                st.info("Thank you for your feedback! We'll use it to improve.")
               
