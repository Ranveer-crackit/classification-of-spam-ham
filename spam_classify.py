!pip install nltk==3.8.1
!pip install scikit-learn
!pip install dill

import pandas as pd

df=pd.read_csv("spam.csv")

df.duplicated(keep=False)

df.duplicated(keep=False).sum()

df.duplicated().sum()
#keeps only first occurence of duplicate value

df1=df.drop_duplicates(keep='first')#will keep the first occurrence of data

#Handle the missing values
df_miss=df1.isnull().sum()

df1.fillna(0)  # Fills all NaN values with 0
df1['Category'].fillna(-1, inplace=True)
   # Fills NaN in 'column_name' with -1, modifying the DataFrame directly


import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
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

df1['processed_text'] = df1['Message'].apply(preprocess_text)

#doing hot encoding by dummy method for classification
dummies=pd.get_dummies(df1.Category)

#concatenate original data frame
merged=pd.concat([df1,dummies],axis='columns')

finaldf=merged.drop(['Category','spam'],axis='columns')
#we can handle with only spam since ham will have true and false value
#final drop and final dataframe on which my model will work
final_df=finaldf.drop(['Message'],axis='columns')


#now for training and testing of the data
from sklearn.model_selection import train_test_split

X=final_df['processed_text']
Y=final_df['ham']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.001,random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
#feature extarction can be needed and cannot be also
#going for bag of words(beer comes 2 times for ham messages and beer=0 for spam message )

# Assuming X_train and X_test are pandas Series containing lists of tokens
vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)#to accept tokens directly

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

#model selection and training
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train_bow,Y_train)

y_predicted=model.predict(X_test_bow)

new_message = "Check out this amazing video!"
processed_message = preprocess_text(new_message)
new_message_bow = vectorizer.transform([processed_message])  # Assuming BoW was used
prediction = model.predict(new_message_bow)

if prediction[0] == 1:
    print("The message is classified as ham.")
else:
    print("The message is classified as spam.")

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_bow)  # Or use X_test_tfidf
accuracy = accuracy_score(Y_test, y_predicted)
report = classification_report(Y_test, y_predicted)

print(f"Accuracy: {accuracy}")
print(report)



import dill
import pickle

with open('vectorizer.pkl', 'wb') as f:
         dill.dump(vectorizer, f)
with open('model.pkl', 'wb') as f:
         pickle.dump(model, f)
