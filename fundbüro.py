import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Klassen
CLASS_NAMES = [
    "T-Shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
    "Sandal", "Hemd", "Sneaker", "Tasche", "Stiefel"
]

st.set_page_config(page_title="Fashion MNIST", layout="centered")
st.title("👕 Fashion-MNIST Erkennung")

# Modell trainieren (einfach & schnell)
@st.cache_resource
def load_model():
    (train_images, train_labels), _ = fashion_mnist.load_data()

    train_images = train_images / 255.0

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_images, train_labels, epochs=3, verbose=0)

    return model

model = load_model()

# Upload
uploaded_file = st.file_uploader("Bild hochladen (schwarz-weiß Kleidung)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Dein Bild", width=200)

    # Preprocessing wie MNIST
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    image = np.array(image) / 255.0

    image = image.reshape(1, 28, 28)

    # Vorhersage
    prediction = model.predict(image)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    st.divider()
    st.subheader(f"Ergebnis: {CLASS_NAMES[index]}")
    st.progress(float(confidence))
    st.write(f"Sicherheit: **{confidence:.2%}**")

else:
    st.info("Bitte ein Bild hochladen.")
