import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, logging
from PIL import Image
import torch
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import threading

# Suppress transformers warnings
logging.set_verbosity_error()
# Lock for thread-safe model use
model_lock = threading.Lock()
@st.cache_resource
def load_model():
        processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False
        )
        model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False
        )
        return processor, model


processor, model = load_model()
st.title("Image Captioning with Nepali Translation and Speech")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or capture from camera")

image_file = uploaded_file or camera_file
if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Selected Image", use_container_width=True)
        st.subheader("Caption:")
        # Generate caption (thread-safe)
        with model_lock:
            inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)


        st.write("English:", caption)
        # Translate to Nepali
        translated = GoogleTranslator(source='en', target='ne').translate(caption)
        st.write("Nepali:", translated)
        # Convert to speech (no temp file)
        tts = gTTS(text=translated, lang="ne")
        mp3_bytes = BytesIO()
        tts.write_to_fp(mp3_bytes)
        mp3_bytes.seek(0)
        st.audio(mp3_bytes.read(), format="audio/mp3")