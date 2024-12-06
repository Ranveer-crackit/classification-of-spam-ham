# classification-of-spam-ham

# Spam Classifier Application

This project is a Machine Learning-based Spam Classifier built using Python. It classifies email messages into **Spam** or **Ham** based on their content. The application is developed with the help of Streamlit for a user-friendly interface and trained using the Naive Bayes classifier on preprocessed text data.

## Features

1. **Text Preprocessing**:
   - Tokenization
   - Stopword Removal
   - Normalization (lowercasing)
   - Stemming and Lemmatization

2. **Machine Learning**:
   - Bag-of-Words (BoW) representation using `CountVectorizer`.
   - Naive Bayes classifier (`MultinomialNB`).

3. **Interactive Interface**:
   - Classify user-entered messages as Spam or Ham.
   - Feedback system to improve the model.

4. **Feedback Recording**:
   - Correct/Incorrect classifications are stored in a `feedback.csv` file for further analysis.
