import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import joblib  # For saving/loading models
import gzip  # For compressing model files

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
name_column = ['id', 'entity', 'target', 'Tweet content']
df = pd.read_csv('twitter_training.csv', names=name_column)

# Drop unnecessary columns and handle missing values
df = df.drop(columns=['id', 'entity'], axis=1)
df.dropna(inplace=True)

# Preprocessing function
ps = PorterStemmer()
stops = set(stopwords.words('english'))

def preprocessing_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    token = text.split()
    token = [ps.stem(word) for word in token if word not in stops]
    return ' '.join(token)

df['Tweet content'] = df['Tweet content'].apply(preprocessing_text)

# Feature extraction
tf = TfidfVectorizer(max_features=5000)
x = tf.fit_transform(df['Tweet content'])
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)

# Build and train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence warnings are encountered
lr_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model and vectorizer using joblib and gzip
model_path = 'sentiment_model.pkl.gz'
with gzip.open(model_path, 'wb') as f:
    joblib.dump(lr_model, f)
joblib.dump(tf, 'vectorizer.pkl.gz')

# Load model and vectorizer for prediction
with gzip.open(model_path, 'rb') as f:
    loaded_model = joblib.load(f)
loaded_vectorizer = joblib.load('vectorizer.pkl')

# User input for sentiment analysis
st.header("Test Sentiment Analysis")
user_input = st.text_area("Enter a Tweet for Sentiment Analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        user_input = preprocessing_text(user_input)
        user_vector = loaded_vectorizer.transform([user_input])
        prediction = loaded_model.predict(user_vector)
        st.write(f"The sentiment is: **{prediction[0]}**")
    else:
        st.write("Please enter a tweet.")
