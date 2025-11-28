<<<<<<< HEAD
import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or paragraph and check whether it's **Fake** or **Real**.")

# Input box
user_input = st.text_area("📝 Enter News Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform and predict
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        if prediction == "FAKE":
            st.error("🚨 The news seems **FAKE**.")
        else:
            st.success("✅ The news seems **REAL**.")
=======
import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or paragraph and check whether it's **Fake** or **Real**.")

# Input box
user_input = st.text_area("📝 Enter News Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform and predict
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        if prediction == "FAKE":
            st.error("🚨 The news seems **FAKE**.")
        else:
            st.success("✅ The news seems **REAL**.")




>>>>>>> 9cf3b77 (Initial commit)
