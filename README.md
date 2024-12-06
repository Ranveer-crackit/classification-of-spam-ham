# Spam Classifier Application

This project is a Machine Learning-based Spam Classifier built using Python. It classifies email messages into **Spam** or **Ham** based on their content. The application is developed with the help of Streamlit for a user-friendly interface and trained using the Naive Bayes classifier on preprocessed text data.


## Overview

The Spam Classifier processes user-input messages, applies a trained Naive Bayes model, and classifies the messages as spam or ham. Key features include:
- **Text Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization.
- **Machine Learning Model**: A Naive Bayes classifier trained on Bag-of-Words (BoW) features.
- **Interactive Interface**: A Streamlit-based UI for easy interaction.
- **Feedback System**: Records user feedback on classification results for future enhancements.

---

## Repository Structure
- ├── app.py                  #Application file
- ├── spam_classify.py         #File which create model
- ├── model.pkl               #Trained Naive Bayes model
- ├── vectorizer.pkl          #CountVectorizer object
- ├── spam.csv                #Original dataset 
- ├── feedback.csv            #Feedback log for user inputs
- ├── README.md               #Project documentation

---
## Deployed on Streamlit
- https://classification-of-spam-ham-uzwrut9xjmcm54xehybab4.streamlit.app/

