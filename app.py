import streamlit as st
from PIL import Image
import tempfile
from predict_image import predict_leaf_disease

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("Plant Leaf Disease Detection")
st.write(
    "Upload a plant leaf image. The system uses image processing features "
    "and machine learning to predict the disease class."
)

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.convert("RGB").save(temp_file.name)
        temp_path = temp_file.name

    if st.button("Predict Disease"):
        try:
            disease, confidence = predict_leaf_disease(temp_path)

            st.success("Prediction Completed")
            st.subheader("Result")
            st.write(f"**Predicted Class:** {disease}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            if "healthy" in disease.lower():
                st.info("The leaf appears healthy according to the model.")
            else:
                st.warning("The model detected a disease category.")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please upload a leaf image.")
