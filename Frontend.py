# importing required libraries
import tensorflow as tf
import numpy as np
import streamlit as st


def remove_digit(text):
    """
    Removes numbers from text
    """
    return "".join(word for word in text if not word.isdigit())


# preprocess and make prediction with the model
def preprocess(text):
    """
    Makes predictions on given text and returns whether it is a real news or not
    """

    # preprocess text
    text = np.array(text)
    custom_text_modified = []
    for i in range(len(text)):
        custom_text_modified.append(remove_digit(text[i]))

    # loading model
    model = tf.keras.models.load_model("final_model_GRU")

    # making predictions
    preds = model.predict(custom_text_modified)

    # rounding off the predictions
    preds = int(np.round(preds))

    # returning the predictions
    if preds == 0:
        return "Real News"

    return "Fake News"


def main():
    st.title("Real and Fake News Detector")
    text = st.text_area("Enter the News you saw:", placeholder="Enter here...",
                        help="Provide the News description you want to verify", height=350, max_chars=10000)

    if st.button("Submit"):
        st.snow()
        if text == "":
            st.warning(" Please Enter some News to verify...", icon="⚠️")

        elif len(text) < 100:
            st.warning(" Please provide more information...", icon="⚠️")

        else:
            result = preprocess([text])
            if result == "Real News":
                st.success(result, icon="✅")
            else:
                st.error(result, icon="❌")

    st.info("This Detector helps you to be aware of the fake news that are being published.", icon="ℹ️")


if __name__ == "__main__":
    # call main function
    main()