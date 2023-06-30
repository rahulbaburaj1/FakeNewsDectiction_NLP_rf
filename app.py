import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re


def transform_text(text):
    y = []
    text = text.lower()
    text = text.split()
    for i in text:
        sc = re.sub('[^a-zA-Z]', ' ', i)
        y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('rf_model.pkl','rb'))

st.title("Fake News Dectiction")

input_data = st.text_input("Enter the Content")

if st.button("Predict"):
    transformed_data = transform_text(input_data)
    v_input = tf.transform([transformed_data])
    result = model.predict(v_input)[0]
    if result == 0:
        st.header("The news is Fake")
    else:
        st.header("The news is Real")
