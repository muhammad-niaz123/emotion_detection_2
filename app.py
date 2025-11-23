import streamlit as st
import os

DATA_DIR = "data"

st.title("Teachable Machine - Create Classes & Upload Images")

class_name = st.text_input("Enter Class Name")

if st.button("Add Class"):
    if class_name:
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_path):

            os.makedirs(class_path)
            st.success(f"Class '{class_name}' created!")
        else:
            st.warning("This class already exists!")


st.subheader("Upload Images for a Class")


if os.path.exists(DATA_DIR):
    classes = os.listdir(DATA_DIR)
else:
    classes = []

if classes:
    selected_class = st.selectbox("Choose Class", classes)

    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)

    if st.button("Save Images"):
        if uploaded_files:
            save_path = os.path.join(DATA_DIR, selected_class)

            for file in uploaded_files:
                file_path = os.path.join(save_path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            st.success(f"{len(uploaded_files)} images saved to class '{selected_class}'!")
        else:
            st.error("Please upload at least one image.")


from trainers.train_logistic_regression import train_logreg_model
from trainers.train_random_forest import train_rf_model
from trainers.train_cnn import train_cnn_model





if st.button("Train Logistic Regression"):
    train_logreg_model()
   
    model, acc, cm, X_test, y_test = train_logreg_model()
    st.success("Logistic Regression trained!")
    st.write("Accuracy:", acc)
    st.write("Confusion Matrix:")
    st.write(cm)





if st.button("Train Random Forest"):
    train_rf_model()
    
    acc, cm, X_test, y_test = train_rf_model()
    st.success("Random Forest trained!")
    st.write("Accuracy:", acc)
    st.write("Confusion Matrix:")
    st.write(cm)

import time
if st.button("Train CNN"):
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    
    for i in range(10):
        time.sleep(0.5)  
        progress_bar.progress((i+1)*10)
        status_text.text(f"Epoch {i+1}/10 running...")

    train_cnn_model()  
    st.success("CNN training complete!")




import joblib
model = joblib.load("models/logreg_model.pkl")

st.header("Make Prediction")

uploaded_img = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])
if uploaded_img:
    from PIL import Image
    import numpy as np

    # img = Image.open(uploaded_img).convert("RGB")
    # img_resized = img.resize((64, 64))
    # img_array = np.array(img_resized).flatten().reshape(1, -1)

    
    # model = joblib.load("models/logreg_model.pkl")

    # pred = model.predict(img_array)[0]

   
    # classes = os.listdir("data") 
    # st.image(img, caption="Uploaded Image")
    # st.success(f"Prediction: {classes[pred]}")

    model_choice = st.selectbox(
    "Select Model for Prediction",
    ["CNN", "Logistic Regression", "Random Forest"]
)

    from tensorflow.keras.models import load_model

    if model_choice == "CNN":
        model = load_model("models/cnn_model.h5")
    elif model_choice == "Logistic Regression":
        model = joblib.load("models/logreg_model.pkl")
    elif model_choice == "Random Forest":
        model = joblib.load("models/rf_model.pkl")

    if st.button("Predict"):

        img = Image.open(uploaded_img).convert("RGB")
        img_resized = img.resize((64, 64))
        classes = os.listdir("data")

        if model_choice == "CNN":

            img_array = np.array(img_resized).reshape(1, 64, 64, 3) / 255.0
            pred = np.argmax(model.predict(img_array))
            st.success(f"CNN Prediction: {classes[pred]}")

        elif model_choice == "Logistic Regression":

            img_array = np.array(img_resized).flatten().reshape(1, -1)
            pred = model.predict(img_array)[0]
            st.success(f"Logistic Regression Prediction: {classes[pred]}")

        elif model_choice == "Random Forest":

            img_array = np.array(img_resized).flatten().reshape(1, -1)
            pred = model.predict(img_array)[0]
            st.success(f"Random Forest Prediction: {classes[pred]}")
