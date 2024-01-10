import streamlit as st
import joblib

def load_model_and_vectorizer():
    # Load both the model and the vectorizer
    loaded_model = joblib.load("random_forest_model.pkl")
    loaded_vectorizer = joblib.load("vectoriser.pkl")
    return loaded_model, loaded_vectorizer
# Add a description/header


# Your Streamlit app code goes here

def main():
    st.image("plst.png", width=200)
    st.title("Multimodal News Detection on Palestinian-Israeli War")
    st.markdown("Welcome to our news verification platform. We analyze news and provide answers on whether the events are true or fake related to the Palestinian-Israeli War.")

    loaded_model, loaded_vectorizer = load_model_and_vectorizer()

    # User input text box
    user_input = st.text_area("Enter text for prediction:")

    if st.button("Predict"):
        # Transform the user input using the loaded vectorizer
        transformed_input = loaded_vectorizer.transform([user_input])

        # Perform prediction using the loaded model
        prediction = loaded_model.predict(transformed_input)[0]
       
        # Display the prediction result and corresponding image
        st.write(f"Predicted Class: {prediction}")
        
        if prediction == 0:
            st.image("true_image.png", width=100, caption="True News")
        else:
            st.image("fake_image.png", width=100, caption="Fake News")

if __name__ == "__main__":
    main()
