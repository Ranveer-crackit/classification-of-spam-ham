import streamlit as st
import dill
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import csv

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
    # Tokenization
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab') 
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')  

    tokens = nltk.word_tokenize(text)
    # Removing Stop Words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    # Normalization (Lowercasing)
    normalized_tokens = [token.lower() for token in filtered_tokens]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in normalized_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]

    return lemmatized_tokens

# Classification function
def classify_text(text):
    probability = model.predict_proba([text])[0][1]  # Probability of spam
    threshold = 0.7
    return "spam" if probability > threshold else "ham"

# Streamlit app
st.title("Spam Classifier")
st.write("This is a application to classify emails as spam or ham.")

# Input fields
message = st.text_area("Enter a message to classify:")
classify_button = st.button("Classify")

# Classification logic
if st.button("Classify"):
    if not message.strip():
        st.error("Please enter a valid message before classifying.")
    else:
        st.session_state.message = message  # Store message in session_state
        
        # Show a spinner while processing
        with st.spinner('Classifying...'):
            processed_message = preprocess_text(message)
            vectorized_message = vectorizer.transform([processed_message])
            prediction = model.predict(vectorized_message)[0]
        if prediction == 1:
            st.success("The message is classified as ham.")
        else:
            st.error("The message is classified as spam.")
        
        # Feedback section
        st.markdown("---")
        st.subheader("Feedback")
        st.write("Your feedback helps us improve our model!")
        feedback='feedback.csv'

        feedback_option = st.radio("Was this classification correct?", ('Correct', 'Incorrect'))
        
        if feedback_option == 'Correct':
            with open(feedback, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([message, prediction, 1])  # 1 for correct
            st.success("Thank you for your feedback!")
        
        elif feedback_option == 'Incorrect':
            with open(feedback, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([message, prediction, 0])  # 0 for incorrect
            st.info("Thank you for your feedback! We'll use it to improve.")
