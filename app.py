# Streamlit UI
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

# Set page configuration
# UI elements
st.markdown("<h1 style='text-align: center; color: white;'>🌿 Guruji Air</h1>", unsafe_allow_html=True)
st.title("🌱 Plant Species Identifier")
st.subheader("Upload, Capture or Live Detect Plant Images")

# Load model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_identifier_model.keras')
    return model

model = load_model()

# Class names
class_names = [
    "African Violet (Saintpaulia ionantha)", "Aloe Vera", "Anthurium (Anthurium andraeanum)",
    "Areca Palm (Dypsis lutescens)", "Asparagus Fern (Asparagus setaceus)", "Begonia (Begonia spp.)",
    "Bird of Paradise (Strelitzia reginae)", "Birds Nest Fern (Asplenium nidus)", "Boston Fern (Nephrolepis exaltata)",
    "Calathea", "Cast Iron Plant (Aspidistra elatior)", "Chinese Money Plant (Pilea peperomioides)",
    "Chinese evergreen (Aglaonema)", "Christmas Cactus (Schlumbergera bridgesii)", "Chrysanthemum",
    "Ctenanthe", "Daffodils (Narcissus spp.)", "Dracaena", "Dumb Cane (Dieffenbachia spp.)",
    "Elephant Ear (Alocasia spp.)", "English Ivy (Hedera helix)", "Hyacinth (Hyacinthus orientalis)",
    "Iron Cross begonia (Begonia masoniana)", "Jade plant (Crassula ovata)", "Kalanchoe",
    "Lilium (Hemerocallis)", "Lily of the valley (Convallaria majalis)", "Money Tree (Pachira aquatica)",
    "Monstera Deliciosa (Monstera deliciosa)", "Orchid", "Parlor Palm (Chamaedorea elegans)",
    "Peace lily", "Poinsettia (Euphorbia pulcherrima)", "Polka Dot Plant (Hypoestes phyllostachya)",
    "Ponytail Palm (Beaucarnea recurvata)", "Pothos (Ivy arum)", "Prayer Plant (Maranta leuconeura)",
    "Rattlesnake Plant (Calathea lancifolia)", "Rubber Plant (Ficus elastica)", "Sago Palm (Cycas revoluta)",
    "Schefflera", "Snake plant (Sanseviera)", "Tradescantia", "Tulip", "Venus Flytrap",
    "Yucca", "ZZ Plant (Zamioculcas zamiifolia)"
]

# Image preprocessing
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img_array = preprocess_input(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction
def predict_frame(frame):
    img_array = preprocess_frame(frame)
    predictions = model.predict(img_array)
    top_pred_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class = class_names[top_pred_idx]
    return predicted_class, confidence

input_method = st.radio("Select Image Source:", ["Upload Image", "Capture from Camera", "Live Detection"])
uploaded_file = None
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Selected Image', use_container_width=True)
        st.write("Predicting...")
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        top_pred_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_class = class_names[top_pred_idx]
        st.success(f"🌿 **Prediction:** {predicted_class}")
        st.info(f"📈 **Confidence:** {confidence*100:.2f}%")

elif input_method == "Capture from Camera":
    uploaded_file = st.camera_input("Take a photo of the plant")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Captured Image', use_container_width=True)
        st.write("Predicting...")
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        top_pred_idx = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_class = class_names[top_pred_idx]
        st.success(f"🌿 **Prediction:** {predicted_class}")
        st.info(f"📈 **Confidence:** {confidence*100:.2f}%")

elif input_method == "Live Detection":
    st.info("Starting Live Plant Detection...")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.predicted_label = None
            self.confidence = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            predicted_class, confidence = predict_frame(img)
            self.predicted_label = predicted_class
            self.confidence = confidence

            display_text = f"{predicted_class} ({confidence*100:.1f}%)"
            font_scale = 0.5 if len(display_text) > 30 else 0.8

            max_width = 40
            lines = [display_text[i:i+max_width] for i in range(0, len(display_text), max_width)]

            overlay = img.copy()
            alpha = 0.4
            y0, dy = 30, 30

            for i, line in enumerate(lines):
                y = y0 + i * dy
                text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_w, text_h = text_size
                cv2.rectangle(overlay, (5, y - text_h - 5), (5 + text_w + 10, y + 5), (0, 0, 0), -1)

            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            for i, line in enumerate(lines):
                y = y0 + i * dy
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(key="plant-live", video_processor_factory=VideoProcessor)

else:
    st.warning("👆 Please upload, capture, or start live detection to identify a plant.")
