import streamlit as st
import dill
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import csv

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

# Streamlit app
st.title("Spam Classifier")
st.write("This is a Machine Learning application to classify emails as spam or ham.")
message = st.text_area("Enter a message:")

if st.button("Classify"):
    processed_message = preprocess_text(message)
    new_message_bow = vectorizer.transform([processed_message])
    prediction = model.predict(new_message_bow)

    if prediction[0] == 1:
        st.success("The message is classified as ham.")
    else:
        st.error("The message is classified as spam.")
     # Feedback section
    st.markdown("---")  # Add a separator
    st.subheader("Feedback")
    col1, col2 = st.columns(2)
    st.write("Your feedback will be used to make our model more strong.Thanks!")
    
    
    with col1:
        if st.button("Correct"):
            # Handle correct feedback (e.g., store in a database or file)
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("Incorrect"):
            # Handle incorrect feedback (e.g., store in a database or file)
            st.write("Thank you for your feedback! We'll use it to improve the classifier.")



# ... (inside the 'if st.button("Correct"):' block) ...
    with open('feedback.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([message, prediction[0], 1])  # 1 for correct

# ... (inside the 'if st.button("Incorrect"):' block) ...
    with open('feedback.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([message, prediction[0], 0])  # 0 for incorrect

