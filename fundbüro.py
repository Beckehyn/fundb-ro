import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Seite konfigurieren
st.set_page_config(page_title="KI Fundbüro (YOLOv8)", layout="centered")
st.title("🔍 KI Fundbüro Erkennung (YOLOv8)")

# Modell laden (YOLOv8)
@st.cache_resource
def load_model():
    # Möglichkeit 1: lokales Modell
    return YOLO("best.pt")
    
    # Möglichkeit 2: von Hugging Face laden
    # return YOLO("hf_hub:username/modelname")

model = load_model()

# Upload
uploaded_file = st.file_uploader("Lade ein Foto hoch...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Foto", use_column_width=True)

    with st.spinner("YOLO analysiert das Bild..."):
        results = model(image)

    # Ergebnisse anzeigen
    st.divider()
    st.subheader("Erkannte Objekte:")

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            st.write(f"**{label}** – Sicherheit: {conf:.2%}")
            st.progress(conf)
    else:
        st.warning("Keine Objekte erkannt.")
else:
    st.info("Bitte lade ein Bild hoch.")
