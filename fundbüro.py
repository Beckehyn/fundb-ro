import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Seite
st.set_page_config(page_title="KI Fundbüro", layout="centered")
st.title("🔍 KI Fundbüro (YOLOv8)")

# Modell laden (Cache!)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # funktioniert immer (Testmodell)

model = load_model()

# Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Dein Bild", use_column_width=True)

    with st.spinner("Analysiere Bild..."):
        results = model(image)

    result = results[0]

    st.divider()
    st.subheader("Erkennung:")

    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            st.write(f"**{label}** ({conf:.2%})")
            st.progress(conf)

        # Bild mit Boxen anzeigen
        annotated = result.plot()
        st.image(annotated, caption="Erkannte Objekte")
    else:
        st.warning("Nichts erkannt 🤷")
else:
    st.info("Bitte ein Bild hochladen.")
