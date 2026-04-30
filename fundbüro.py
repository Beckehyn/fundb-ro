import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

CLASS_NAMES = [
    "T-Shirt/Top", "Hose", "Pullover", "Kleid", "Mantel",
    "Sandal", "Hemd", "Sneaker", "Tasche", "Stiefel"
]

st.set_page_config(page_title="Fashion MNIST", layout="centered")
st.title("👕 Fashion-Erkennung (verbessert)")

# ✅ Besseres CNN-Modell
@st.cache_resource
def load_model():
    (train_images, train_labels), _ = fashion_mnist.load_data()

    train_images = train_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(train_images, train_labels, epochs=5, verbose=0)

    return model

model = load_model()

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Original", width=200)

    # 🔥 WICHTIG: Preprocessing verbessern
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)

    # Kontrast erhöhen
    image = ImageOps.autocontrast(image)

    # Optional invertieren (MNIST ist oft hell auf dunkel)
    image = ImageOps.invert(image)

    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    st.image(image.reshape(28,28), caption="Verarbeitet", width=200)

    prediction = model.predict(image)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    st.divider()
    st.subheader(f"Ergebnis: {CLASS_NAMES[index]}")
    st.progress(float(confidence))
    st.write(f"Sicherheit: **{confidence:.2%}**")

else:
    st.info("Bitte ein Bild hochladen.")
